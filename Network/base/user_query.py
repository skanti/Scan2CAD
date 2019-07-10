from enum import Enum 
import os
import shutil
import pathlib
import glob
import torch
import re
import JSONHelper

class MODE(Enum):
    NEW = 0
    REPLACE = 1
    LOAD = 2

class UserQuery:
    def setup(self, model, output, projname=""):
        counter_iteration = 0
        print(projname)
        mode, projectdir = self.menu(output, projname)
        logfile = projectdir + "/log_eval.json"
        if mode == MODE.NEW:
            print("MODE: NEW")
            pathlib.Path(projectdir).mkdir(parents=True, exist_ok=True)
            self.create_log(logfile)
        elif mode == MODE.REPLACE:
            print("MODE: REPLACE")
            shutil.rmtree(projectdir, ignore_errors=True)
            pathlib.Path(projectdir).mkdir(parents=True, exist_ok=True)
            self.create_log(logfile)
        elif mode == MODE.LOAD:
            print("MODE: LOAD")
            checkpointfile, counter_iteration = self.load_most_recent_checkpoint(projectdir, "/cp-*")
            print("Current iteration:", counter_iteration)
            if os.path.exists(logfile):
                self.clip_log_by_number(logfile, counter_iteration)
            else:
                self.create_log(logfile)

            self.load_parts_of_checkpoint(checkpointfile, model)
        self.make_backup_of_important_files(projectdir)

        return projectdir, logfile, counter_iteration

    def make_backup_of_important_files(self, projectdir):
        shutil.copy2("./model.py", projectdir + "/model_backup.py")
        shutil.copy2("./main.py", projectdir + "/main_backup.py")

    def load_parts_of_checkpoint(self, checkpointfile, model):
        model_dict = model.state_dict()
        model_pretrained = torch.load(checkpointfile)

        model_pretrained = {k: v for k, v in model_pretrained.items() if k in model_dict}
        model_dict.update(model_pretrained) 

        model.load_state_dict(model_dict)

    
    def create_log(self, logfile):
        JSONHelper.write(logfile, [])

    def check_inputed_name(self, basedir, projname):
        projectdir = basedir + "/" + projname
        if len(glob.glob(projectdir + "/cp-*")) > 0:
            return MODE.LOAD, projectdir
        else:
            return MODE.NEW, projectdir

    def menu(self, basedir, projname):
        if projname == "":
            return self.query_user(basedir)
        else:
            return self.check_inputed_name(basedir, projname)

    def query_user(self, basedir):
        text = "dummy"
        try:
            text = input("\nEnter project name: ")
        except:
            quit()

        if text == "":
            text = self.create_folder_increment(basedir, "/run")

        projectdir = basedir + "/" + text
        print("project-dir:", projectdir)
        if not os.path.exists(projectdir):
            return MODE.NEW, projectdir
        else:
            text = ""
            try:
                text = input("Exist already. Replace [r] or Load [l]?: ")
            except:
                quit()
            if text == "r":
                return MODE.REPLACE, projectdir
            elif text == "l":
                return MODE.LOAD, projectdir
        quit()
    def create_folder_increment(self, basedir, foldername0):
        for i in range(1000):
            suffix = foldername0 + str(i)
            foldername = basedir + "/" + suffix
            if not os.path.exists(foldername):
                return suffix

    def extract_number(self, f):
        s = re.findall("(\d+).pth",f)
        return (int(s[0]) if s else -1,f)

    def load_most_recent_checkpoint(self, dir, wildcard):
        files = glob.glob(dir + wildcard)
        files = [self.extract_number(f) for f in files]
        files = sorted(files, key=lambda x : x[0])
        res = files[-1]
        return res[1], res[0]

    def clip_log_by_number(self, logfile, num_max):
        data = JSONHelper.read(logfile)
        clipped = [item for item in data if item["iteration"] <= num_max]
        JSONHelper.write(logfile, clipped)

