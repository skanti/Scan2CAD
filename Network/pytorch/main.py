import sys
assert(sys.version_info >= (3, 5))

import argparse
import pathlib
import os, inspect, time
import sys
import csv
import random
import torch
import torch.utils.data as torchdata
import numpy as np
import model as modelnn
#import model_skip as modelnn

sys.path.append("../base")
import user_query
import kernels
import sample_loader
import HeatmapStatistics
import IsMatchStatistics
import dataloader
import losses
import SaveOutput
import JSONHelper


# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--name', default="", help="project name")
parser.add_argument('--weight', type=float, default=8, help='input batch size')
parser.add_argument('--train_list', required=True, help='path to file list of h5 train data')
parser.add_argument('--val_list', default="", help='path to file list of h5 val data')
parser.add_argument('--visual_list', default="", help="data to be visualized (saved out)")
parser.add_argument('--output', default="./output/", help='folder to output model checkpoints and predictions')
# train params
parser.add_argument('--mask_neg', type=int, required=True, default=1, help='mask negative samples?')
parser.add_argument('--with_match', type=int, required=True, default=1, help='train with negative samples?')
parser.add_argument('--with_scale', type=int, required=True, default=1, help='train with scale prediction?')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--max_iteration', type=int, default=int(1e6), help='number of iteration to train for')
parser.add_argument('--n_samples_eval', type=int, default=int(1e6), help='num-max to evaluate of val set')
parser.add_argument('--interval_eval', type=int, default=2000, help='interval for evaluation')
parser.add_argument('--n_threads', type=int, default=4, help="number of threads")
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--seed', type=int, default=-1, help='seed for rng')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--just_visualize', type=bool, default=False, help='no training and evluation, just feed forward')

opt = parser.parse_args()

# data files
assert os.path.exists(opt.train_list), "train-list does not exists. Maybe data not generated?"
train_list = JSONHelper.read(opt.train_list)
opt.val_list = opt.train_list if opt.val_list == "" else opt.val_list
val_list = JSONHelper.read(opt.val_list)
opt.visual_list = opt.val_list if opt.visual_list == "" else opt.visual_list
visual_list = JSONHelper.read(opt.visual_list)
print(opt)

gkern3d_dim32 = torch.FloatTensor(torch.from_numpy(kernels.gaussian3d(7, 1.5))).view(1, 1, 7, 7, 7).cuda()

model = modelnn.Model3d().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=0.0005)
n_model_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))


if opt.seed == -1:
    opt.seed = time.time()

torch.manual_seed(opt.seed)
if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed) 



class TrainingWorld:
    def __init__(self, counter_iteration, projectdir, logfile):
        self.counter_iteration = counter_iteration
        self.projectdir = projectdir
        self.logfile = logfile
        self.metafile = self.projectdir + "/metadata.json"

    def write_metadata(self):
        metadata = {"num-samples" : len(self.dataset_train), "batch-size" : opt.batch_size, "lr" : opt.lr, "weight" : opt.weight, "n-model-params": n_model_params}
        JSONHelper.write(self.metafile, metadata)

    def start(self):
        self.dataset_train = dataloader.DatasetOnlineLoad(train_list)
        self.write_metadata()
        self.timing = time.time()

        self.iterations_per_epoch = len(train_list)//opt.batch_size
        if opt.interval_eval < 0:
            opt.interval_eval = abs(opt.interval_eval)*self.iterations_per_epoch

        self.counter_offset = self.counter_iteration
        last_epoch = self.counter_offset//self.iterations_per_epoch if self.counter_offset > 0 else -1

        milestones = [int(1e5 + i*3e4) for i in range(10)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5, last_epoch=-1)
        for i in range(self.counter_offset):
            self.scheduler.step()
        for param_group in optimizer.param_groups:
            print("lr", param_group["lr"])
        while self.counter_iteration < (self.counter_offset + opt.max_iteration):
            #scheduler.step()
            self.train()
    
    def train(self):
        model.train()
        
        self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads, drop_last=True)

        # ----------------------------------------------------------------------
        for i_batch, samples_batch in enumerate(self.dataloader_train):
            self.scheduler.step()
            if self.counter_iteration % opt.interval_eval == 0:
                print("timing", time.time() - self.timing)
                self.save_checkpoint()
                with torch.no_grad():
                    self.evaluate()
                    self.visualize()
                self.timing = time.time()
                model.train()

            sys.stdout.write("\rtraining i-iteration: %i / %d         " % (self.counter_iteration, opt.max_iteration + self.counter_offset))
            sys.stdout.flush()

            losses_batch, outputs = self.feed_forward(samples_batch)
            
            loss_batch = torch.FloatTensor([0]).cuda()
            for k,v in losses_batch.items():
                loss_batch += v[0]*v[1] # <- loss*weight

            self.backprop(loss_batch)
            
            self.counter_iteration += 1

    def save_checkpoint(self):
        torch.save(model.state_dict(), os.path.join(projectdir, "cp-%d.pth" % self.counter_iteration))

    def backprop(self, loss_batch):
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        
    def feed_forward(self, samples):
        in0 = samples["sdf_scan"].cuda()
        in0 = torch.nn.ConstantPad3d((0, 1, 0, 1, 0, 1), -0.15)(in0)
        in0[in0 > 1.8] = 1.8
        in1 = samples["df_cad"].cuda()
        heatmap_gt = samples["heatmap"].cuda()
        match_gt = samples["match"].type(torch.FloatTensor).cuda()
        scale_gt = samples["scale"].type(torch.FloatTensor).cuda()

        mask = match_gt.clone() if opt.mask_neg else torch.ones(match_gt.shape).cuda()
        
        heatmap_gt = torch.nn.functional.conv3d(heatmap_gt, gkern3d_dim32, padding=3)
        
        heatmap0, match0, scale0 = model(in0, in1)

        heatmap1 = torch.zeros(heatmap0.shape).cuda()
        heatmap2 = torch.zeros(heatmap0.shape).cuda()
        heatmap3 = torch.zeros(heatmap0.shape).cuda()
        scale1 = torch.zeros(scale0.shape).cuda()
        
        for i in range(opt.batch_size):
            if mask[i]:
                scale1[i] = scale0[i]
                heatmap1[i] = torch.nn.Sigmoid()(heatmap0[i])
                heatmap2[i] = torch.nn.functional.log_softmax(heatmap0[i].view(-1), dim=0).view(1, 32, 32, 32)
                # -> for evalulation
                heatmap3[i] = torch.nn.functional.softmax(heatmap0[i].view(-1), dim=0).view(1, 32, 32, 32)
                heatmap3[i] = torch.mul(heatmap1[i]/torch.max(heatmap1[i]), heatmap3[i]/torch.max(heatmap3[i]))
                # <-

        loss = {}
        output = {}

        loss0 = losses.weighted_bce_heatmap(heatmap1, heatmap_gt, weight=opt.weight, batch_mask=mask)
        loss1 = losses.weighted_cross_entropy_heatmap(heatmap2, heatmap_gt, weight=opt.weight, batch_mask=mask)

        loss["heatmap"] = (loss0 + loss1, 1.0)
        #loss["heatmap"] = torch.FloatTensor([0]).cuda()

        if opt.with_match:
            match0 = torch.nn.Sigmoid()(match0)
            loss["match"] = (losses.weighted_bce(match0.view(-1), match_gt.view(-1)), 0.1)
            #if opt.mask_neg == False:
            #    for i in range(opt.batch_size):
            #        heatmap3[i] *= match0[i]
        else:
            match0 *= 0.0
            for i in range(opt.batch_size):
                match0[i] = torch.clamp(torch.sum(heatmap1[i])/100.0, 0, 1)
                heatmap3[i] *= 1.0 if match0[i] > 0.5 else 0.0

        if opt.with_scale:
            loss["scale"] = (losses.mse(scale1, scale_gt, batch_mask=mask), 0.2)
        else:
            scale1 *= 0.0


        output["heatmap"] = heatmap3.data
        output["heatmap_gt"] = heatmap_gt
        output["match"] = match0.data
        output["match_gt"] = match_gt.data
        output["scale"] = scale1.data
        output["scale_gt"] = scale_gt.data
        
        return loss, output


    def evaluate(self):
        model.eval()
        
        # --------------------------------------------------------------------------
        def run(source_list, n_max_samples=-1):
            stats_heatmap = HeatmapStatistics.Statistics()
            stats_match = IsMatchStatistics.Statistics()
            self.dataset = dataloader.DatasetOnlineLoad(source_list, n_max_samples=n_max_samples)
            self.dataloader = torchdata.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_threads, drop_last=True)

            losses_container = {}
            for i_batch, samples_batch in enumerate(self.dataloader):
                sys.stdout.write("\reval: i_batch: %i / %d             " % (i_batch, self.dataset.n_samples//opt.batch_size))
                sys.stdout.flush()
                
                losses, outputs = self.feed_forward(samples_batch)

                stats_heatmap.update(outputs["heatmap"], outputs["heatmap_gt"])
                stats_match.update(outputs["match"], outputs["match_gt"])
                for k, v in losses.items():
                    losses_container.setdefault(k, []).append(v[0].item())
            # ---------------------------------------------------------------

            outputs = {}
            outputs["f1_heatmap"] = stats_heatmap.f1()
            outputs["f1_ismatch"] = stats_match.f1()

            pr = {}
            pr["heatmap"] = stats_heatmap.pr_curve()
            pr["ismatch"] = stats_match.pr_curve()

            losses_avg = {}
            for k, v in losses_container.items():
                losses_avg[k] = np.mean(v)
            return losses_avg, outputs, pr
        
        print("\n##########################")
        losses_avg_train, outputs_train, pr_train = run(train_list, min(opt.n_samples_eval, len(train_list)))
        print("\ntrain-loss:", losses_avg_train)
        print("train-eval:", outputs_train)
        
        losses_avg_val, outputs_val, pr_val = run(val_list, min(opt.n_samples_eval, len(val_list)))
        print("\nval-loss:", losses_avg_val)
        print("val-eval:", outputs_val)
        print("##########################")

        losses_avg = {"train" : losses_avg_train, "val" : losses_avg_val}
        outputs = {"train" : outputs_train, "val" : outputs_val}
        data = {"iteration" : self.counter_iteration, "loss" : losses_avg, "eval" : outputs} 

        data0 = JSONHelper.read(self.logfile)
        data0.append(data)
        JSONHelper.write(self.logfile, data0)

        with open(projectdir + "/pr_curve_heatmap.csv", "w") as csvfile:
            for row in pr_val["heatmap"]:
                csv.writer(csvfile, delimiter=',').writerow(row)
        
        with open(projectdir + "/pr_curve_ismatch.csv", "w") as csvfile:
            for row in pr_val["ismatch"]:
                csv.writer(csvfile, delimiter=',').writerow(row)

    def visualize(self):
        model.eval()
        
        # --------------------------------------------------------------------------
        self.dataset_visual = dataloader.DatasetOnlineLoad(visual_list)
        self.dataloader_visual = torchdata.DataLoader(self.dataset_visual, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_threads, drop_last=True)
        for i_batch, samples_batch in enumerate(self.dataloader_visual):
            sys.stdout.write("\rvisualize: i_batch: %i / %d " % (i_batch, self.dataset_visual.n_samples//opt.batch_size))
            sys.stdout.flush()
            losses_batch, outputs = self.feed_forward(samples_batch)
            SaveOutput.save_output(opt.batch_size, projectdir, samples_batch, outputs)
        # ---------------------------------------------------------------

if __name__ == '__main__':
    counter_iteration = 0

    user_query = user_query.UserQuery()
    projectdir, logfile, counter_iteration = user_query.setup(model, opt.output, opt.name)

    world = TrainingWorld(counter_iteration, projectdir, logfile)

    if opt.just_visualize == True:
        with torch.no_grad():
            world.visualize()
    else:
        world.start()
