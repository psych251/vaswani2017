import pandas as pd
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import copy
import pickle
import tokenizer as tk
import random
import skimage.io
from crab.utils import get_max_key

def rand_sample(arr, n_samples=1):
    """
    Randomly samples a single element from the argued array.

    arr: sequence of some sort
    """
    if not isinstance(arr,list): arr = list(arr)
    if len(arr) == 0: print("len 0:", arr)
    samples = []
    perm = np.random.permutation(len(arr))
    for i in range(n_samples):
        samples.append(arr[perm[i]])
    if len(samples) == 1: return samples[0]
    return samples

class WordProblems(Dataset):
    def __init__(self, difficulty="easy", lowercase=True,
                                          n_samples=10000,
                                          max_count=1000,
                                          split_digits=False,
                                          verbose=True,
                                          index=True,
                                          use_structs=True,
                                          samp_ps={"start":.15,
                                                   "end":.7,
                                                   "move":.15},
                                          **kwargs):
        """
        This is a class to assist in automatic generation of word
        problems. Each entry consists of a problem setup with a
        question. All entries and answers are strings.
        
        Examples:
        x = "you move {n1} {obj_type1} to the goal, then you move
        {n2} {obj_type2} to the goal, ... 
        How many objects are at the goal?"
        y = "{n1+n2+...} objects"

        difficulty: str
            the difficulty of the dataset
        lowercase: bool
            if true, all letters are lowercase
        n_samples: int
            the number of samples for the dataset
        max_count: int
            the maxium number for the word problems to use
        split_digits: bool
            split digits into individual 0-9 tokens
        index: bool
            if false, the sentence based questions and answers are
            not tokenized and converted to indices.
        use_structs: bool
            if true, uses a list of dictionaries of sample questions
            broken into their sentence parts with the corresponding
            count dicts for each action to build questions and answers
            on the fly.
        samp_ps:dict
            keys: str
                the types of samples
            vals: float
                the corresponding proportion of samples that should be
                this type of sample
        """
        self.tokenizer = tk.Tokenizer()
        self.lowercase = lowercase
        self.n_samples = n_samples
        self.max_count = max_count
        self.samp_ps = samp_ps
        self.use_structs = use_structs
        self.split_digits = split_digits
        rand_labels = False
        if difficulty == "rand" or difficulty == "random":
            rand_labels = True
            difficulty = "easy"
            

        self.attrs = ["color", "shape", "all", "none"]
        self.colors=["red","blue","green","yellow","orange","purple"]
        self.shapes = ["box", "ball", "cylinder", "coin", "ramp"]
        self.obj_types = []
        for color in self.colors:
            for shape in self.shapes:
                self.obj_types.append((color+" "+shape).strip())

        self.initial_conditions = [
            "<count> <type> objects start at <loc>",
            "<count> <type> objects are initialized at <loc>",
            #"you move <count> <type> objects to <loc>",
            #"you take <count> <type> objects to <loc>",
            #"you put <count> <type> objects at <loc>",
            ]

        self.action_statements = [
            "you <verb> <count> <type> objects <preposition1> "+\
                                 "<loc1> <preposition2> <loc2>"
            ]

        self.verbs =  ["move", "take"]
        self.qverbs = ["are at", "started at", "ended at"]
                  #"existed at some point at"]
        self.pos_prepositions = ["to"]
        self.neg_prepositions = ["from", "away from"]
        self.prepositions = [self.pos_prepositions,self.neg_prepositions]
        self.locations = ["the grey area","the brown area"]

        self.directions = {
            "move":["in front of","left of","right of","on top of",
                     "behind"], 
            "remove": ["from"],
            "take":["from", "away from"]}

        if verbose:
            print("Making data")
        samples = self.make_data_set(n_samples=n_samples,
                                     lowercase=lowercase,
                                     difficulty=difficulty,
                                     max_count=max_count,
                                     samp_ps=self.samp_ps,
                                     use_structs=self.use_structs,
                                     verbose=verbose)
        self.samp_structs = samples[-1]
        self.sampled_types = self.count_obj_types(self.samp_structs)
        samples = [list(zip(*s)) for s in samples[:-1]]
        start_samps, end_samps, move_samps = samples
        self.start_samps = start_samps # list of lists
        self.end_samps = end_samps
        self.move_samps = move_samps
        questions = start_samps[0]+end_samps[0]+move_samps[0]
        answers = start_samps[1]+end_samps[1]+move_samps[1]

        self.samples = {"start":start_samps, "end":end_samps,
                                             "move":move_samps}
        self.questions = questions
        self.answers = answers
        extras = self.colors+self.shapes
        self.tokenizer = tk.Tokenizer(X=questions,
                                   Y=answers,
                                   split_digits=self.split_digits,
                                   index=False,
                                   prepend=True,
                                   strings=extras,
                                   append=True)
        if index:
            xlen = self.tokenizer.seq_len_x
            self.X = self.tokenizer.index_tokens(self.tokenizer.token_X,
                                                 prepend=True,
                                                 append=True,
                                                 seq_len=xlen)
            self.tokenizer.X = self.X
            ylen = self.tokenizer.seq_len_y
            self.Y = self.tokenizer.index_tokens(self.tokenizer.token_Y,
                                                 prepend=True,
                                                 append=True,
                                                 seq_len=ylen)
            if rand_labels:
                for i in range(len(self.Y)):
                    self.Y[i][1:3]=torch.randint(len(self.idx2word),(2,))
            self.tokenizer.Y = self.Y
            self.questions = self.tokenizer.string_X
            self.answers = self.tokenizer.string_Y
            self.token_qs = self.tokenizer.token_X
            self.token_ans = self.tokenizer.token_Y

    def struct_getitem(self,idx):
        """
        Use this function to sample from the sentence structs to create
        sentences on the fly.

        idx: int
            the index of the struct
        """
        q,a = self.struct2qa(self.samp_structs[idx])
        t = self.tokenizer
        tok_q,_,_ = t.tokenize([q])
        tok_a,_,_ = t.tokenize([a])
        x = self.tokenizer.index_tokens(tok_q,self.seq_len_x,
                                                prepend=True,
                                                append=True,
                                                verbose=False)
        y = self.tokenizer.index_tokens(tok_a,self.seq_len_y,
                                                prepend=True,
                                                append=True,
                                                verbose=False)
        return x[0],y[0]

    def struct2qa(self, struct, qtype="rand"):
        """
        This function takes in a single struct and returns a
        complete question-answer pair.

        struct: dict
            items:
                "init": dict()
                    "string":the actual initialization string
                    "counts": corresponding count dict
                "actions": dict()
                    "strings": list of the action strings
                    "counts": list of the corresponding count dict
                        of each action and all actions leading up to
                        this point.
                    "infos": list of the corresponding info dict
                        of each action        
        qtype: str ("end", "start", "move", "rand", "random")
            the type of question. available options are "end", "start",
            "move", "rand". "rand" will return a random question type
            with probability corresponding to the self.samp_ps dict
        """
        options = {"end":self.get_end_qs,
                   "start":self.get_start_qs,
                   "move":self.get_move_qs}
        sentence = struct["init"]["string"]
        for i,action in enumerate(struct["actions"]["strings"]):
            if i > 0:
                sentence += "then "
            sentence += action
        count_dict = struct['actions']["counts"][-1]
        count_dict["start"] = struct["init"]["counts"]["end"]
        if qtype == "rand" or qtype == "random":
            keys = list(options.keys())
            keys,ps = list(zip(*self.samp_ps.items()))
            qtype = np.random.choice(keys,p=ps)
        if qtype == "end":
            qas = self.get_end_qs(sentence, count_dict)
        elif qtype == "start":
            qas = self.get_start_qs(sentence, count_dict)
        elif qtype == "move":
            qas = self.get_move_qs(sentence, count_dict)
        return random.sample(qas, 1)[0]

    def count_obj_types(self, structs):
        """
        Searches the structs dict to find every object type (color
        shape combination) used in the data. Returns a dict of these
        object types with the number of data samples each object type
        appeared in.

        structs: list of count dicts
            each entry in the list corresponds to a sample. The count
            dicts have the following structure
            keys: string
                "init": dict
                    keys: str
                        "counts": dict
                            keys: str
                                "end": dict
                                    keys: str
                                        "<loc>": dict
                                            keys: str
                                                "all": dict
                                                    keys: str
                                                       "<obj_type>":int
        Returns:
            all_types: dict
                A dict of every object type as keys with the number
                of samples that the object type appears in as the
                corresponding value.
                keys: str
                    object types
                values: int
                    the number of samples in which this object type
                    appears
        """
        all_types = {ot:0 for ot in self.obj_types}
        for s,struct in enumerate(structs):
            obj_types = set()
            locations = struct["init"]["counts"]["end"]
            for loc in locations.keys():
                for obj_type in locations[loc]["all"].keys():
                    obj_types.add(obj_type)
            for obj_type in obj_types:
                all_types[obj_type] += 1
        return all_types

    @property
    def inits(self):
        return self.tokenizer.inits
    @property
    def word2idx(self):
        return self.tokenizer.word2idx
    @property
    def idx2word(self):
        return self.tokenizer.idx2word
    @property
    def MASK(self):
        return self.tokenizer.MASK
    @property
    def START(self):
        return self.tokenizer.START
    @property
    def STOP(self):
        return self.tokenizer.STOP
    @property
    def seq_len_x(self):
        return self.tokenizer.seq_len_x
    @property
    def seq_len_y(self):
        return self.tokenizer.seq_len_y
                        
    def __len__(self):
        if self.use_structs:
            return len(self.samp_structs)
        return len(self.X)

    def __getitem__(self,idx):
        if self.use_structs:
            return self.struct_getitem(idx)
        return self.X[idx],self.Y[idx]

    def get_init(self, count_dict, max_count=10,
                                   min_count=1,
                                   obj_types=None,
                                   locs=None,
                                   n_inits=1,
                                   allow_dups=True):
        """
        Creates an inital condition

        count_dict: dict
            keys: str
                moved, end, start
                these are the potential things to keep track of. start
                tracks the objects that started at different locations.
                end tracks object counts as they are moved around the
                different locations. moved tracks the total times an
                object type is moved.
            vals: dict
                keys: str
                    the locations or the object types for "moved"
                vals: dict
                    keys: str
                        the object types
                    vals: int
                        the counts of the type at that location
        max_count: int
            the maximum possible number of objects to be initialized.
        min_count: int
            the minimum possible number of objects to be initialized.
        obj_types: sequence or None
            this is an optional argument that can be used to restrict
            the setting to particular object types. This is useful if
            you want a particularly difficult problem.
        locs: sequence or None
            this is an optional argument that can be used to restrict
            the setting to particular locations.
        n_inits: int
            if you want to instantiate multiple objects in one go.
        allow_dups: bool
            if true, then objects of the same type can be initialized
            mulitple times at the same location.
        """
        if n_inits <= 0: n_inits = 1
        if max_count is None: max_count = 10
        if min_count is None: min_count = 1
        if locs is not None: loc = rand_sample(locs)
        else: loc = rand_sample(self.locations)
        condition = rand_sample(self.initial_conditions)
        if n_inits > 1:
            condition = condition.replace("<count> <type>",
                                          "<count1> <type1>")
        for i in range(n_inits):
            count = rand_sample(range(min_count, max_count+1))

            if obj_types is not None:
                obj_type = rand_sample(obj_types)
            else:
                obj_types = set(self.obj_types)
                if not allow_dups:
                    for obj in count_dict["start"][loc]["all"].keys():
                        if obj in obj_types:
                            obj_types.remove(obj)
                try:
                    obj_type = rand_sample(list(obj_types))
                except:
                    print("Duplicating obj types due to insufficient types")
                    obj_type = rand_sample(self.obj_types)

            color,shape = [o.strip() for o in obj_type.split(" ")]
            count_dict["start"][loc]["all"][obj_type] += count
            count_dict["start"][loc]["color"][color] += count
            count_dict["start"][loc]["shape"][shape] += count
            count_dict["start"][loc]["none"] += count
            count_dict["end"][loc]["all"][obj_type] += count
            count_dict["end"][loc]["color"][color] += count
            count_dict["end"][loc]["shape"][shape] += count
            count_dict["end"][loc]["none"] += count
            if "you move" in condition or "you take" in condition or\
                    "you put" in condition:
                count_dict["moved"]["all"][obj_type] += count
                count_dict["moved"]["color"][color] += count
                count_dict["moved"]["shape"][shape] += count
                count_dict["moved"]["none"] += count
                
            keys = ["<count>","<type>","<loc>"]
            vals = [count, obj_type, loc]
            info = {k:v for k,v in zip(keys,vals)}
            if i < n_inits-2:
                condition = condition.replace("<count1> <type1>",
                              "<count> <type>, <count1> <type1>")
                for k,v in info.items():
                    condition = condition.replace(k,str(v))
            elif i < n_inits-1:
                condition = condition.replace("<count1> <type1>",
                              "<count> <type>, and <count1> <type1>")
                for k,v in info.items():
                    condition = condition.replace(k,str(v))
            else:
                condition = condition.replace("<count1> <type1>",
                                              "<count> <type>")
                for k,v in info.items():
                    condition = condition.replace(k,str(v))

        if count == 1 and n_inits == 1:
            condition = condition.replace("objects", "object")
            condition = condition.replace("start ", "starts ")
            condition = condition.replace("are ", "is ")
        return condition, count_dict, info

    def get_action(self, count_dict, start_loc=None,
                                     end_loc=None,
                                     obj_types=None):
        """
        Creates an action statement

        count_dict: dict
            keys: str
                moved, end, start
                these are the potential things to keep track of. start
                tracks the objects that started at different locations.
                end tracks object counts as they are moved around the
                different locations. moved tracks the total times an
                object type is moved.
            vals: dict
                keys: str
                    the locations or the object types for "moved"
                vals: dict
                    keys: str
                        the object types
                    vals: int
                        the counts of the type at that location
        obj_types: sequence or None
            this is an optional argument that can be used to restrict
            the action to particular object types. This is useful if
            an initialization condition was used or if you want a
            particularly difficult problem.
        """
        if start_loc is None or end_loc is None:
            keys = count_dict["end"].keys()
            start_loc,end_loc = rand_sample(keys,2)
        objs = []
        objs_was_none = obj_types is None
        while len(objs) == 0:
            objs = []
            counts = count_dict["end"][start_loc]["all"]
            if objs_was_none:
                obj_types = set(counts.keys())
            for obj in counts.keys():
                if counts[obj] > 0 and obj in obj_types:
                    objs.append(obj)
            if len(objs) == 0:
                keys = count_dict["end"].keys()
                start_loc,end_loc = rand_sample(keys,2)
        assert start_loc != end_loc
        obj_type = rand_sample(objs)
        count = rand_sample(list(range(counts[obj_type]+1)))
        prep1 = rand_sample(self.neg_prepositions)
        prep2 = rand_sample(self.pos_prepositions)
        verb = rand_sample(self.verbs)
        color,shape = [o.strip() for o in obj_type.split(" ")]
        count_dict["end"][start_loc]["all"][obj_type] -= count
        count_dict["end"][start_loc]["color"][color] -= count
        count_dict["end"][start_loc]["shape"][shape] -= count
        count_dict["end"][start_loc]["none"] -= count
        count_dict["end"][end_loc]["all"][obj_type] += count
        count_dict["end"][end_loc]["color"][color] += count
        count_dict["end"][end_loc]["shape"][shape] += count
        count_dict["end"][end_loc]["none"] += count
        count_dict["moved"]["all"][obj_type] += count
        count_dict["moved"]["color"][color] += count
        count_dict["moved"]["shape"][shape] += count
        count_dict["moved"]["none"] += count
        info = {"type":obj_type,
                "count":count,
                "to":end_loc,
                "from":start_loc}
        # For variety
        if np.random.random() > .5:
            prep1,prep2 = prep2,prep1
            start_loc,end_loc = end_loc,start_loc

        statement = rand_sample(self.action_statements)
        if count == 1: statement=statement.replace("objects", "object")
        keys = ["<verb>", "<count>", "<type>","<preposition1>",
                "<loc1>", "<preposition2>", "<loc2>"]
        vals = [verb, str(count), obj_type, prep1, start_loc,
                                                   prep2, end_loc]
        for k,v in zip(keys,vals):
            statement = statement.replace(k, v)
        return statement, count_dict, info

    def get_count_dict(self):
        """
        Returns a dictionary that is compatible with the make_data_set
        func. The returned dict has the following structure:

        {
            "start":{
                "loc1":{
                    "color":{
                        "red":0,
                        "blue":...
                    },
                    "shape":{
                        "cylinder":0,
                        "box":...
                    },
                    "all":{
                        "red cylinder":0,
                        "red box":...
                    }
                    "none":0 # sum of all objects
                },
                "loc2":...
            },
            "end":{
                "loc1":{
                    "color":{
                        "red":0,
                        "blue":...
                    },
                    "shape":{
                        "cylinder":0,
                        "box":...
                    },
                    "all":{
                        "red cylinder":0,
                        "red box":...
                    }
                    "none":0 # sum of all objects
                },
                "loc2":...
            }
            "moved":{
                "color":{
                    "red":0,
                    "blue":...
                },
                "shape":{
                    "cylinder":0,
                    "box":...
                },
                "all":{
                    "red cylinder":0,
                    "red box":...
                }
                "none":0 # sum of all objects
            }
        """
        def zero(): return 0
        attrs = self.attrs
        moved_dict = {attr:defaultdict(zero) for attr in attrs}

        for obj in self.obj_types:
            color, shape = obj.split(" ")
            moved_dict["color"][color.strip()] = 0
            moved_dict["shape"][shape.strip()] = 0
            moved_dict["all"][obj] = 0
            moved_dict["none"] = 0
        count_dict ={
            "start":{l:{attr:defaultdict(zero) for attr in attrs}\
                                         for l in self.locations},
            "end":  {l:{attr:defaultdict(zero) for attr in attrs}\
                                         for l in self.locations},
            "moved":moved_dict
            }
        for loc in self.locations:
            count_dict["start"][loc]["none"] = 0
            count_dict["end"][loc]["none"] = 0
        return count_dict

    def make_data_set(self, n_samples=5000, lowercase=True,
                                            difficulty="easy",
                                            max_count=1000,
                                            use_structs=False,
                                            samp_ps={"start":.15,
                                                     "end":.7,
                                                     "move":.15},
                                            verbose=True):
        """
        Makes an easy dataset. Numbers and answers are always 5 or
        less. The form of the statements is always 1 sub-statement
        maximum. 

        n_samples: int
            the number of training examples
        lowercase: bool
            if true, all letters are lowercase
        difficulty: str
            easy, medium or hard
        use_structs: bool
            if true, samples are counted based off of the number of
            structs instead of the number of samples
        samp_ps:dict
            keys: str
                the types of samples
            vals: float
                the corresponding proportion of samples that should be
                this type of sample


        Returns:
            <generic>_samples: set of (q,a) tuples
                each sample set contains string question answer tuples
            samp_structs: list of dicts
                each dict has two keys:
                    "init": dict()
                        "string":the actual initialization string
                        "counts":corresponding count dict
                    "actions": dict()
                        "strings": list of the action strings
                        "counts": list of the corresponding count dict
                            of each action and all actions leading up to
                            this point.
                        "infos": list of the corresponding info dict
                            of each action        
        """
        samp_structs = []
        start_samples = set()
        end_samples = set()
        move_samples = set()
        sentences = set()
        samples = [start_samples, end_samples, move_samples]
        samp_dict = {"start":start_samples, "end":end_samples,
                                            "move":move_samples}
        samp_len = 0
        n_loops = 0
        while samp_len < n_samples or n_loops>(n_samples+100000):
            n_loops += 1
            samp_struct = {"init":{"string":None, "counts":None},
                           "actions":{ "strings": [],
                                       "counts": [],
                                       "infos":  [] } }
            count_dict = self.get_count_dict()

            # Initial conditions
            sentence = ""
            if difficulty.lower() == "easy":
                n_inits = np.random.randint(1,3)
                allow_dups = False
            elif difficulty.lower() == "medium":
                n_inits = np.random.randint(2,5)
                allow_dups = True
            else:
                n_inits = np.random.randint(1,10)
                allow_dups = True
            start_idx,end_idx = 0,1
            if np.random.random() > .5:
                start_idx,end_idx = 1,0

            # Initial Condition Loop
            while n_inits > 0:
                rand_amt = np.random.randint(1,n_inits+1)
                init_condish,count_dict,info= self.get_init(count_dict,
                                                  max_count=max_count,
                                                  min_count=1,
                                                  locs=self.locations,
                                                  n_inits=rand_amt,
                                                  allow_dups=allow_dups)
                n_inits -= rand_amt
                sentence += init_condish
                if n_inits > 0: sentence += " and "
            sentence += ". "
            samp_struct['init']["string"] = sentence
            temp_dict = copy.deepcopy(count_dict)
            del temp_dict["start"]
            samp_struct['init']["counts"] = temp_dict

            if difficulty.lower() == "easy":
                n_inits = np.random.randint(1,3)
            elif difficulty.lower() == "medium":
                n_inits = np.random.randint(2,6)
            else:
                n_inits = np.random.randint(5,10)

            # Action Statement Loop
            for i in range(n_inits):
                tup = self.get_action(count_dict=count_dict,
                                      start_loc=None,
                                      end_loc=None)
                action,count_dict,info = tup
                action = action + ". "
                sentence += action
                if i < n_inits-1: sentence += "then "
                samp_struct['actions']["strings"].append(action)
                temp_dict = copy.deepcopy(count_dict)
                del temp_dict["start"]
                samp_struct['actions']["counts"].append(temp_dict)
                samp_struct['actions']["infos"].append(info)

            if sentence not in sentences:
                sentences.add(sentence)
                samp_structs.append(samp_struct)
                start_samples |= self.get_start_qs(sentence, count_dict)
                end_samples |= self.get_end_qs(sentence, count_dict)
                move_samples |= self.get_move_qs(sentence, count_dict)
            
            if use_structs:
                samp_len = len(samp_structs)
            else:
                max_key = get_max_key(samp_ps)
                samp_len = len(samp_dict[max_key])/samp_ps[max_key] 
            if verbose:
                print(round(samp_len/float(n_samples)*100), "%",
                                                      end="     \r")

        start_len=min(int(samp_len*samp_ps["start"]),len(start_samples))
        start_samples = random.sample(start_samples, start_len)
        move_len=min(int(samp_len*samp_ps["move"]), len(move_samples))
        move_samples = random.sample(move_samples, move_len)
        end_len = min(int(samp_len*samp_ps["end"]), len(end_samples))
        end_samples = random.sample(end_samples, end_len)
        if verbose:
            s = "Sample Portions:\nstart: {}\nend: {}\nmove: {}"
            print(s.format(start_len,end_len,move_len))
        return start_samples, end_samples, move_samples, samp_structs
    
    def get_start_qs(self, sentence, count_dict):
        """
        Returns a question along with an answer that corresponds to the
        current struct and count dict.

        sentence: str
            the current sentence
        count_dict: dict
            the corresponding count dict to this sentence (see
            get_count_dict() for more information on the structure
            of a count_dict)
        """
        samples = set()
        for loc in count_dict["start"].keys():
            s = "how many <type> objects started at {}?"
            question = s.format(loc)
            for attr in self.attrs:
                if attr == "none":
                    q = question.replace("<type> ","")
                    q = sentence + q
                    count = count_dict["start"][loc][attr]
                    ans = "{} objects".format(count)
                    if count == 1: ans = ans[:-1]
                    q = q.lower()
                    ans = ans.lower()
                    samples.add((q,ans))
                    continue
                for obj in count_dict["start"][loc][attr].keys():
                    q = question.replace("<type>",obj)
                    q = sentence + q
                    count = count_dict["start"][loc][attr][obj]
                    ans = "{} {} objects".format(count,obj)
                    if count == 1: ans = ans[:-1]
                    q = q.lower()
                    ans = ans.lower()
                    samples.add((q,ans))
        return samples

    def get_end_qs(self, sentence, count_dict):
        """
        Returns a question along with an answer that corresponds to the
        current struct and count dict.

        sentence: str
            the current sentence
        count_dict: dict
            the corresponding count dict to this sentence (see
            get_count_dict() for more information on the structure
            of a count_dict)
        """
        samples = set()
        for loc in self.locations:
            for attr in self.attrs:
                s = "how many <type> objects are at {}?".format(loc)
                if attr == "none":
                    q = s.replace("<type> ", "")
                    q = sentence + q
                    count = count_dict["end"][loc][attr]
                    ans = "{} objects".format(count)
                    if count == 1: ans = ans[:-1]
                    q = q.lower()
                    ans = ans.lower()
                    samples.add((q,str(ans)))
                    continue
                for obj in count_dict["end"][loc][attr].keys():
                    q = s.replace("<type>",obj)
                    q = sentence + q
                    count = count_dict["end"][loc][attr][obj]
                    ans = "{} {} objects".format(count,obj)
                    if count == 1: ans = ans[:-1]
                    q = q.lower()
                    ans = ans.lower()
                    samples.add((q,str(ans)))
        return samples

    def get_move_qs(self, sentence, count_dict):
        """
        Returns a question along with an answer that corresponds to the
        current struct and count dict.

        sentence: str
            the current sentence
        count_dict: dict
            the corresponding count dict to this sentence (see
            get_count_dict() for more information on the structure
            of a count_dict)
        """
        samples = set()
        s = "how many <type> objects did you move?"
        for attr in self.attrs:
            if attr == "none":
                q = s.replace("<type> ","")
                q = sentence + q
                count = count_dict["moved"][attr]
                ans = "{} objects".format(count)
                if count == 1: ans = ans[:-1]
                q = q.lower()
                ans = ans.lower()
                samples.add((q,ans))
                continue
            objs = set()
            for loc in count_dict["start"].keys():
                objs |= set(count_dict["start"][loc][attr].keys())
            for obj in objs:
                q = s.replace("<type>",obj)
                q = sentence + q
                count = count_dict["moved"][attr][obj]
                ans = "{} {} objects".format(count,obj)
                if count == 1: ans = ans[:-1]
                q = q.lower()
                ans = ans.lower()
                samples.add((q,ans))
        return samples

class ListWrapper:
    def __init__(self, arr, shape=None):
        """
        arr: some list
        shape: a tuple that does not include the len of the arr
        """
        self.arr = arr
        self._shape = tuple(shape)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self,idx):
        if isinstance(idx,int):
            return self.arr[idx]
        else:
            temp = []
            for i in idx:
                temp.append(self.arr[i])
            return temp

    @property
    def shape(self):
        shape = tuple([len(self.arr)]+list(self._shape))
        return shape

def get_data(seq_len=10, shuffle_split=False,
                         train_p=0.8,
                         dataset="Journal",
                         n_samples=50000,
                         difficulty="medium",
                         split_digits=False,
                         max_count=1000,
                         **kwargs):
    """
    Returns two torch Datasets, one validation and one training.

    seq_len: int
        the length of word sequences
    dataset: str
        the name of the dataset to be used
    shuffle_split: bool
        if true, shuffles before the split
    train_p: float (0,1)
        the portion of data reserved for training
    n_samples: int
        some datasets allow a specifications of the quantity of data
        points
    split_digits: bool
        split digits into individual 0-9 tokens
    """

    if "end_p" in kwargs:
        samp_ps = {"end":kwargs["end_p"],
                   "start":kwargs["start_p"],
                   "move":kwargs["move_p"]}
    dataset_name = dataset
    dataset = globals()[dataset](seq_len=seq_len,
                                 difficulty=difficulty,
                                 n_samples=n_samples,
                                 split_digits=split_digits,
                                 max_count=max_count,
                                 samp_ps=samp_ps,
                                 split="train",
                                 **kwargs)
    if dataset_name == "CLEVR":
        val_dataset = globals()[dataset_name](seq_len=seq_len,
                                 difficulty=difficulty,
                                 n_samples=n_samples,
                                 split_digits=split_digits,
                                 max_count=max_count,
                                 samp_ps=samp_ps,
                                 split="val",
                                 word2idx=dataset.word2idx,
                                 idx2word=dataset.idx2word,
                                 **kwargs)
        return dataset, val_dataset
    split_idx = int(len(dataset)*train_p)
    if len(dataset)-split_idx > 30000: split_idx = len(dataset)-30000
    if shuffle_split:
        perm = torch.randperm(len(dataset)).long()
    else:
        if dataset_name=="WordProblems":
            print("WARNING!!!! shuffle_split is off but is highly recommended for WordProblems datasets.\nPLEASE STOP THIS TRAINING SESSION AND TURN SHUFFLE SPLIT ON")
        perm = torch.arange(len(dataset)).long()
    train_idxs = perm[:split_idx]
    val_idxs = perm[split_idx:]

    val_dataset = DatasetWrapper(dataset,val_idxs)
    dataset = DatasetWrapper(dataset,train_idxs)

    return dataset, val_dataset

class DatasetWrapper(Dataset):
    """
    Used as a wrapper class to more easily split a dataset into a
    validation and training set
    """
    def __init__(self,dataset,idxs):
        """
        dataset: torch Dataset
        idxs: torch LongTensor or list of ints
        """
        self.dataset = dataset
        for attr in dir(dataset):
            if "__"!=attr[:2]:
                try:
                    setattr(self, attr, getattr(dataset,attr))
                except:
                    pass
        self.idxs = idxs
        assert len(self.idxs) <= len(self.dataset)
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        try:
            idx = self.idxs[idx]
            return self.dataset[idx]
        except FileNotFoundError as e:
            while True:
                try:
                    idx = rand_sample(self.idxs)
                    return self.dataset[idx]
                except FileNotFoundError as e:
                    pass

