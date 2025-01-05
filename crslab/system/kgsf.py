# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/3
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os

import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class KGSFSystem(BaseSystem):
    """This is the system for KGSF model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KGSFSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        batch = [ele.to(self.device) for ele in batch]
        if stage == 'pretrain':
            info_loss = self.model.forward(batch, stage, mode)
            if info_loss is not None:
                self.backward(info_loss.sum())
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == 'rec':
            rec_loss, info_loss, rec_predict = self.model.forward(batch, stage, mode)
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss.sum())
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.sum().item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
            if info_loss:
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.forward(batch, stage, mode)
                if mode == 'train':
                    self.backward(gen_loss.sum())
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.sum().item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.forward(batch, stage, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Pretrain epoch {str(epoch)}]')
            for batch in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=False):
                self.step(batch, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
            self.model.freeze_parameters()
        else:
            if hasattr(self.model, "module"):
                self.model.module.freeze_parameters()
            else:
                self.model.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.pretrain()
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        """
        KGSFSystem 상호작용(inference) 메서드 예시 코드.
        - 사용자 입력: self.get_input()
        - 토큰화/엔티티링크: self.tokenize(), self.link()
        - context 관리: self.update_context()
        - 추천(rec) + 대화(conv) 단계 순서대로 forward
        """

        # (1) 대화 인터랙션 초기화
        self.init_interact()
        print("=== Start interacting with KGSF ===")
        self.model.eval()  # 추론 모드

        language = 'en'  # 혹은 self.language

        while not self.finished:
            user_input = self.get_input(language=language)
            if self.finished:
                print("Bye!")
                break

            # 2) 토큰화 + 엔티티/단어 링크
            tokens = self.tokenize(user_input, 'nltk')
            entity_list = self.link(tokens, self.side_data['entity_kg']['entity'])
            word_list   = self.link(tokens, self.side_data['word_kg']['entity'])

            # 3) vocab 인덱싱
            token_ids = [self.vocab['tok2ind'].get(t, self.vocab['unk']) for t in tokens]
            entity_ids = [
                self.vocab['entity2id'][ent] for ent in entity_list
                if ent in self.vocab['entity2id']
            ]
            movie_ids = [eid for eid in entity_ids if eid in self.item_ids]
            word_ids = [
                self.vocab['word2id'][w] for w in word_list
                if w in self.vocab['word2id']
            ]

            # 4) context 갱신
            self.update_context('rec', token_ids=token_ids, entity_ids=entity_ids,
                                item_ids=movie_ids, word_ids=word_ids)
            self.update_context('conv', token_ids=token_ids, entity_ids=entity_ids,
                                item_ids=movie_ids, word_ids=word_ids)

            # (A) 추천 단계
            context_entities = torch.LongTensor(self.context['rec']['context_entities']).unsqueeze(0).to(self.device)
            context_words    = torch.LongTensor(self.context['rec']['context_words']).unsqueeze(0).to(self.device)
            if len(entity_ids) == 0:
                entity_ids_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
            else:
                entity_ids_tensor = torch.LongTensor([entity_ids[0]]).to(self.device)

            # 추론 시 CrossEntropyLoss를 계산할 필요가 없으므로 dummy label
            movie_ids_tensor = torch.zeros(1, dtype=torch.long, device=self.device)

            batch_rec = [
                context_entities,
                context_words,
                entity_ids_tensor,
                movie_ids_tensor
            ]

            with torch.no_grad():
                # forward()가 (rec_loss, info_loss, rec_scores)를 반환한다!
                rec_loss, info_loss, rec_scores = self.model.forward(batch_rec, stage='rec', mode='infer')

                # rec_scores is shape (1, n_entity). Now we can do rec_scores.cpu()[0]
                rec_scores = rec_scores.cpu()[0]
                rec_scores = rec_scores[self.item_ids]
                _, rank = torch.topk(rec_scores, 5, dim=-1)
                recommended_items = [self.item_ids[idx] for idx in rank.tolist()]
                print("[Recommend]:", recommended_items)

            # (B) 대화 단계
            context_tokens_list = []
            for toks in self.context['conv']['context_tokens']:
                context_tokens_list.extend(toks)
            if not context_tokens_list:
                context_tokens_list = [self.vocab['start']]

            context_tokens = torch.LongTensor(context_tokens_list).unsqueeze(0).to(self.device)
            conv_entities  = torch.LongTensor(self.context['conv']['context_entities']).unsqueeze(0).to(self.device)
            conv_words     = torch.LongTensor(self.context['conv']['context_words']).unsqueeze(0).to(self.device)

            # 테스트 모드 => dummy response
            dummy_response = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            batch_conv = [context_tokens, conv_entities, conv_words, dummy_response]

            with torch.no_grad():
                # forward()가 test 모드에서는 단일 preds 반환
                conv_pred = self.model.forward(batch_conv, stage='conv', mode='test')
                conv_pred = conv_pred.tolist()[0]
                p_str = ind2txt(conv_pred, self.ind2tok, self.end_token_idx)
                print(f"[Response]: {p_str}")