import torch
from neuralkg.eval_task import *
from .BaseLitModel import BaseLitModel
from neuralkg.utils.tools import logging, log_metrics

class indGNNLitModel(BaseLitModel):

    def __init__(self, model, args):
        super().__init__(model, args)

    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser
    
    def training_step(self, batch, batch_idx):

        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        pos_label = batch["positive_label"]
        neg_label = batch["negative_label"]
        
        # pos_score = self.model((pos_sample, pos_label))
        # neg_score = self.model((neg_sample, neg_label))
        pos_score = self.model(pos_sample)
        neg_score = self.model(neg_sample)
        loss = self.loss(pos_score, neg_score)
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)

        logging.info("Train|loss: %.4f at epoch %d" %(loss, self.current_epoch+1))  #TODO: 把logging改到BaseLitModel里面
        return loss
    
    def validation_step(self, batch, batch_idx):

        results = dict()
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        pos_label = batch["graph_pos_label"]
        neg_label = batch["graph_neg_label"]
        
        # score_pos = self.model(pos_sample)
        # score_neg = self.model(neg_sample)
        score_pos = self.model(pos_sample[0])
        score_neg = self.model(neg_sample[0])

        results['pos_scores'] = score_pos.squeeze(1).detach().cpu().tolist()
        results['neg_scores'] = score_neg.squeeze(1).detach().cpu().tolist()
        results['pos_labels']  = pos_label
        results['neg_labels']  = neg_label
        return results
    
    def validation_epoch_end(self, results) -> None:
        outputs = self.get_auc(results, "Eval")
        # self.log("Eval|mrr", outputs["Eval|mrr"], on_epoch=True)
        if self.current_epoch!=0:
            logging.info("++++++++++++++++++++++++++start validating++++++++++++++++++++++++++")
            log_metrics(self.current_epoch+1, outputs)
            logging.info("++++++++++++++++++++++++++over validating+++++++++++++++++++++++++++")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)
    
    def test_step(self, batch, batch_idx):

        # results = dict()
        # ranks = link_predict(batch, self.model, prediction='ind')
        # results["count"] = torch.numel(ranks)
        # results["mrr"] = torch.sum(1.0 / ranks).item()
        # for k in self.args.calc_hits:
        #     results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        # return results

        results = dict()
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        pos_label = batch["graph_pos_label"]
        neg_label = batch["graph_neg_label"]
        
        # score_pos = self.model(pos_sample)
        # score_neg = self.model(neg_sample)
        score_pos = self.model(pos_sample[0])
        score_neg = self.model(neg_sample[0])

        results['pos_scores'] = score_pos.squeeze(1).detach().cpu().tolist()
        results['neg_scores'] = score_neg.squeeze(1).detach().cpu().tolist()
        results['pos_labels']  = pos_label
        results['neg_labels']  = neg_label
        return results


    def test_epoch_end(self, results) -> None:
        # outputs = self.get_results(results, "Test")

        # logging.info("++++++++++++++++++++++++++start testing++++++++++++++++++++++++++")
        # log_metrics(self.current_epoch+1, outputs)
        # logging.info("++++++++++++++++++++++++++over testing+++++++++++++++++++++++++++")

        # self.log_dict(outputs, prog_bar=True, on_epoch=True)

        outputs = self.get_auc(results, "Test")
        # self.log("Eval|mrr", outputs["Eval|mrr"], on_epoch=True)
        if self.current_epoch!=0:
            logging.info("++++++++++++++++++++++++++start Test++++++++++++++++++++++++++")
            log_metrics(self.current_epoch+1, outputs)
            logging.info("++++++++++++++++++++++++++over Test+++++++++++++++++++++++++++")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.   
        """
        milestones = int(self.args.max_epochs) #NOTE: what
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
