import os,shutil,sys
sys.path.append('../')
import pandas as pd
import numpy as np
from copy import deepcopy
import tensorflow as tf
import ast
import json

from utils.custom_metric import recall_avg,precision_m,f1_m
from utils.tools import plot_confusion_matrix,draw_STT_png

from sklearn.metrics import precision_recall_fscore_support,classification_report,confusion_matrix

def check_dir(out_dir):
    import shutil
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)    

def check_dir_no_del(out_dir):
    # import shutil
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)    

    if not os.path.exists(out_dir):
        # shutil.rmtree(out_dir)
        os.makedirs(out_dir)    

def check_dir_filepath(path_in):
    out_dir = os.path.dirname(path_in)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
        
def list2onehot(label_list,n_class):
    val_label = []

    for labels in label_list:
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        onehot_init = [0 for i in range(n_class)]

        if not isinstance(labels, list):
            onehot_init[labels] = 1
        else:
            for idx in labels:
                onehot_init[idx] = 1
        val_label.append(onehot_init)
    return np.array(val_label)

def getBestepoch(checkpoint_dir):
    epoch_list = os.listdir(checkpoint_dir)
    values,epochs=[],[]
    if len(epoch_list) <= 0:
        assert 1>2, checkpoint_dir+' no files'
    for epoch_name in epoch_list:
        epoch_path = deepcopy(epoch_name)
        if '.h5' in epoch_name:
            filename=epoch_name.replace('.h5','')
            v = float(filename.split('_')[-1])
            e = filename.split('_')[1]
            values.append(v)
            epochs.append(e)
    idx = values.index(max(values))
    path = epoch_list[idx]
    best_epoch = epochs[idx]
    return path,best_epoch

def softmax_to_onehot(props):
    '''
    softmax to onehot
    input :
    [[0.2,0.3,0.5],
     [0.7,0.3,0.5],
     [0.7,0.9,0.5]
    ]
    output:
     [[0. 0. 1.]
      [1. 0. 0.]
      [0. 1. 0.]]

    '''
    # Y = tf.one_hot(tf.argmax(t, dimension = 1), depth = 2)

    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

class eval_multi_clss():
    def __init__(self):
        pass

    def _get_keyword_r(self,cfg):
        df=pd.read_csv(cfg.keyword_csv)
        self.labels = df['name'].tolist()
        return df

    def _softmax_to_onehot(self,props):
        # Y = tf.one_hot(tf.argmax(t, dimension = 1), depth = 2)
        if isinstance(props, list):
            props = np.array(props)
        a = np.argmax(props, axis=1)
        b = np.zeros((len(a), props.shape[1]))
        b[np.arange(len(a)), a] = 1
        return b.astype(int)

    def calc(self,cfg,y_true,y_pred,best_epoch='1',out_name=None):
        y_pred = self._softmax_to_onehot(y_pred)

        keyword_df=pd.read_csv(cfg.keyword_csv)
        self.labels = keyword_df['name'].tolist()
        
        # keyword_df = self._get_keyword(cfg)
        # print('y_true',y_true)
        # print('y_pred',y_pred)
        # print('self.labels',self.labels)
        print(classification_report(y_true, y_pred))

        prf1s = precision_recall_fscore_support(y_true, y_pred, average=None)
        precisions = prf1s[0]
        recalls = prf1s[1]
        f1s = prf1s[2]
        categor_p,categor_r,categor_f1 = [],[],[]
        for i in range(cfg.n_classes):
            categor_p.append(precisions[i])
            categor_r.append(recalls[i])
            categor_f1.append(f1s[i])
        
        keyword_df['precision']=categor_p
        keyword_df['recall']=categor_r
        keyword_df['f1']=categor_f1

        if cfg.output_dir and not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)

        if out_name is None:
            out_path = cfg.output_dir+'/e{}_categories_eval_results.csv'.format(best_epoch)
        else:
            out_path = cfg.output_dir+'/e{}_categories_eval_results_{}.csv'.format(best_epoch,out_name)

        keyword_df.to_csv(out_path)
        print(out_path,' saved')

        y_true_idx,y_pred_idx = [],[]
        for y_true_i in y_true.tolist():
            y_true_idx.append(y_true_i.index(max(y_true_i)))

        for y_pred_max_i in y_pred.tolist():
            y_pred_idx.append(y_pred_max_i.index(max(y_pred_max_i)))
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        save_path_norm = cfg.output_dir+'/'+'e'+str(best_epoch)+'_confusion_matrix_normize.png'
        save_path_values = cfg.output_dir+'/'+'e'+str(best_epoch)+'confusion_matrix_values.png'
        plot_confusion_matrix(cm,self.labels,save_path_norm,title='Confusion matrix norm',cmap=None,normalize=True)
        plot_confusion_matrix(cm,self.labels,save_path_values,title='Confusion matrix values',cmap=None,normalize=False)

    def _get_keyword(self,cfg):
        df=pd.read_csv(cfg.keyword_csv)
        clsfications = df['name'].tolist()
        n_alls = df['all'].tolist()
        info_dict = {}
        for i in range(len(clsfications)):
            cls_name = clsfications[i]
            n_all = n_alls[i]
            # if n_all >0:
            info_dict[cls_name]={'right':0,'over':0,'miss':0}
        return df,info_dict,clsfications

    def _p_r_f1(self,n_right,n_miss,n_over,c_type='p'):
        if c_type == 'p':
            if n_right+n_over>0:
                cls_precision = n_right/(n_right+n_over)
                cls_precision = cls_precision*100
                cls_precision = round(cls_precision,3)
            else:
                cls_precision = 'no_pred'
            return cls_precision
        
        elif c_type == 'r':
            if n_right+n_miss>0:
                cls_recall = n_right/(n_right+n_miss)
                cls_recall = cls_recall*100
                cls_recall = round(cls_recall,3)
            else:
                cls_recall = 'no_label'
            return cls_recall

        elif c_type == 'f1':
            cls_p=self._p_r_f1(n_right,n_miss,n_over,c_type='p')
            cls_r=self._p_r_f1(n_right,n_miss,n_over,c_type='r')
            
            if cls_p =='no_pred' or cls_r=='no_label':
                cls_f1 = 'miss_p/r'
            else:
                if cls_p ==0 and cls_r==0:
                    cls_p = 1e-8
                cls_f1 = (2*cls_p*cls_r)/(cls_p+cls_r)
                cls_f1 = round(cls_f1,3)
            return cls_f1
    
    def analysis(self,cfg,y_true,y_pred,file_names,best_epoch='1'):
        def read_stt_json(file_path):
            with open(file_path, 'r') as j:
                # sign_lists = []
                contents = json.loads(j.read())
            return contents['value']
            
        def move2fail(file_name,out_path,pred_name):
            file_namer = os.path.basename(file_name)
            filename, file_extension = os.path.splitext(file_namer)
            
            if file_extension == '.png':
                out_path_pred = '{}{}__{}{}'.format(out_path,filename,pred_name,file_extension)
                shutil.copy2(file_name,out_path_pred)
            elif file_extension == '.json':
                crop_sig = read_stt_json(file_name)
                out_syn_png_path_i = out_path+filename+'.png'
                print('out_syn_png_path_i',out_syn_png_path_i)
                draw_STT_png(crop_sig,out_syn_png_path_i)
                if not os.path.exists(out_syn_png_path_i):
                    print('not found',out_syn_png_path_i)
            else:
                print('unrecrogrize file_extension',file_extension)
                assert 1>2
                
        true_idx = np.argmax(y_true, axis=1)
        pred_idx = np.argmax(y_pred, axis=1)

        out_dir = './model_outputs_analysis_draw/'+cfg.try_time+'/'
        check_dir(out_dir)
        key_df,info_dict,clsfications = self._get_keyword(cfg)

        real_names = list(info_dict.keys())
        for i in range(len(file_names)):
            file_name = file_names[i]
            true_idxi = true_idx[i]
            pred_idxi = pred_idx[i]

            true_name = clsfications[true_idxi]
            pred_name = clsfications[pred_idxi]
            def ST_cheat_rule(true_name,pred_name,clsfications):
                if pred_name=='ST段抬高' and true_name=='ST段弓背样抬高':
                    return True
                else:
                    return False

            ST_cheat_results = ST_cheat_rule(true_name,pred_name,clsfications)

            if true_idxi == pred_idxi or ST_cheat_results is True:
                label_i = real_names[true_idxi]
                info_dict[label_i]['right']+=1
                out_path = '{}{}/right/'.format(out_dir,label_i)
                check_dir_no_del(out_path)
                if len(os.listdir(out_path)) <50:
                    move2fail(file_name,out_path,pred_name)
            else:
                true_label = real_names[true_idxi]
                false_label = real_names[pred_idxi]
                info_dict[true_label]['miss']+=1
                info_dict[false_label]['over']+=1

                out_path_miss = '{}{}/miss/'.format(out_dir,true_label)
                out_path_over = '{}{}/over/'.format(out_dir,false_label)
                check_dir_no_del(out_path_miss)
                check_dir_no_del(out_path_over)

                move2fail(file_name,out_path_miss,pred_name)
                move2fail(file_name,out_path_over,'标_'+true_name)

        n_allright,n_allmiss,n_allover=0,0,0
        n_rightall,n_missall,n_overall=[],[],[]
        cls_ps,cls_rs,cls_f1s=[],[],[]
        for cls_name in clsfications:
            cls_dict = info_dict[cls_name]
            n_right = cls_dict['right']
            n_miss = cls_dict['miss']
            n_over = cls_dict['over']
            
            n_allright+=n_right
            n_allmiss+=n_miss
            n_allover+=n_over

            cls_ps.append(self._p_r_f1(n_right,n_miss,n_over,c_type='p'))
            cls_rs.append(self._p_r_f1(n_right,n_miss,n_over,c_type='r'))
            cls_f1s.append(self._p_r_f1(n_right,n_miss,n_over,c_type='f1'))

            n_rightall.append(n_right)
            n_missall.append(n_miss)
            n_overall.append(n_over)

        cls_allp = self._p_r_f1(n_allright,n_allmiss,n_allover,c_type='p')
        cls_allr = self._p_r_f1(n_allright,n_allmiss,n_allover,c_type='r')
        cls_allf1 = self._p_r_f1(n_allright,n_allmiss,n_allover,c_type='f1')
        
        key_df['p_{0:.4f}'.format(cls_allp)] = cls_ps
        key_df['r_{0:.4f}'.format(cls_allr)] = cls_rs
        key_df['f1_{0:.4f}'.format(cls_allf1)] = cls_f1s
        key_df['got']=n_rightall
        key_df['miss']=n_missall
        key_df['over']=n_overall
        out_path1 = cfg.output_dir+'/e{}_pred_anaysi.csv'.format(best_epoch)
        out_path2 = out_dir+'/e{}_pred_anaysi.csv'.format(best_epoch)
        key_df.to_csv(out_path1)
        key_df.to_csv(out_path2)

class eval_results():
    def __init__(self):
        pass

    def _get_keyword(self,cfg):
        df=pd.read_csv(cfg.keyword_csv)
        names = df['name'].tolist()
        self.name2indx = {name: i for i, name in enumerate(names)}
        self.idx2name = {i: name for i, name in enumerate(names)}
        return df

    def calc_recall(self,y_true,y_pred,index=-1):
        '''
        index: index of label
        input shape:  [n_samples,n_class]
        pred: [[0.2,0.4,0.1,0.5],
                [0.2,0.4,0.1,0.5],
                [0.2,0.4,0.1,0.5]]

        label: [[1,0,0,1],
                [0,1,0,1],
                [1,0,1,1]]
        '''
        if index < 0:
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
            possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        else:
            true_positives = np.sum(np.round(np.clip(y_true[:,index] * y_pred[:,index], 0, 1)))
            possible_positives = np.sum(np.round(np.clip(y_true[:,index], 0, 1)))
        recall = true_positives / (possible_positives + 1e-07)
        n_miss = possible_positives-true_positives
        n_label = possible_positives
        return round(recall,3),int(n_miss),int(n_label)

    def calc_precision(self,y_true,y_pred,index=-1):
        if index < 0:
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
            predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
        else:
            true_positives = np.sum(np.round(np.clip(y_true[:,index] * y_pred[:,index], 0, 1)))
            predicted_positives = np.sum(np.round(np.clip(y_pred[:,index], 0, 1)))
        precision = true_positives / (predicted_positives + 1e-07)
        n_got = true_positives
        n_over = predicted_positives-true_positives
        return round(precision,3),int(n_got),int(n_over)

    def calc_f1(self,y_true,y_pred):
        recall,_,_ = self.calc_recall(y_true, y_pred)
        precision,_,_ = self.calc_precision(y_true, y_pred)
        return round(2*((precision*recall)/(precision+recall+1e-07)),4)

    def _tran2tf(self,y_true,y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred) #pred_org pred  float64
        y_true = tf.dtypes.cast(y_true, tf.float32)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
        return y_true,y_pred

    def calc(self,cfg,y_true,y_pred,best_epoch='1',out_name=None):
        y_true_tf,y_pred_tf = self._tran2tf(y_true,y_pred)

        recall_a = recall_avg(y_true_tf,y_pred_tf)
        recall_a=recall_a.numpy()
        pres_a = precision_m(y_true_tf,y_pred_tf)
        pres_a=pres_a.numpy()
        f1_a = f1_m(y_true_tf,y_pred_tf)
        f1_a=f1_a.numpy()

        keyword_df = self._get_keyword(cfg)
        categor_p,categor_r,categor_f1 = [],[],[]
        categor_label,categor_got,categor_miss,categor_over = [],[],[],[]
        r_avg,_,_ = self.calc_recall(y_true,y_pred)
        p_avg,_,_ = self.calc_precision(y_true,y_pred)
        f1_avg = self.calc_f1(y_true,y_pred)
        for i in range(cfg.n_classes):
        # for i in range(104):
            r_i,n_miss,n_label = self.calc_recall(y_true,y_pred,index=i)
            p_i,n_got,n_over = self.calc_precision(y_true,y_pred,index=i)
            f1_i = round(2*((p_i*r_i)/(p_i+r_i+1e-07)),3)

            categor_p.append(p_i)
            categor_r.append(r_i)
            categor_f1.append(f1_i)

            categor_got.append(n_got)
            categor_miss.append(n_miss)
            categor_over.append(n_over)
            categor_label.append(n_label)

        # keyword_df['p_{0:.2f}'.format(p_avg)]=categor_p
        # keyword_df['r_{0:.2f}'.format(r_avg)]=categor_r
        # keyword_df['f1_{0:.2f}'.format(f1_avg)]=categor_f1

        # print('categor_p',categor_p)
        # print('keyword_df',len(keyword_df))
        keyword_df['p_{0:.3f}_tf{0:.3f}'.format(p_avg,pres_a)]=categor_p
        keyword_df['r_{0:.3f}_tf{0:.3f}'.format(r_avg,recall_a)]=categor_r
        keyword_df['f1_{0:.3f}_tf{0:.3f}'.format(f1_avg,f1_a)]=categor_f1
        keyword_df['label']=categor_label
        keyword_df['got']=categor_got
        keyword_df['miss']=categor_miss
        keyword_df['over']=categor_over
        if cfg.output_dir and not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        
        if out_name is None:
            out_path = cfg.output_dir+'/e{}_categories_eval_results.csv'.format(best_epoch)
        else:
            out_path = cfg.output_dir+'/e{}_categories_eval_results_{}.csv'.format(best_epoch,out_name)

        keyword_df.to_csv(out_path)
        print(out_path,' saved')


class eval_normal_binary():
    def __init__(self):
        pass
    
    def _cm_calc(self,y_true_normal, y_pred_normal):
        prf1s = precision_recall_fscore_support(y_true_normal, y_pred_normal, average=None)
        precisions = prf1s[0]
        recalls = prf1s[1]
        f1s = prf1s[2]
        categor_p,categor_r,categor_f1 = [],[],[]
        for i in range(2):
            categor_p.append('{0:.4f}'.format(precisions[i]))
            categor_r.append('{0:.4f}'.format(recalls[i]))
            categor_f1.append('{0:.4f}'.format(f1s[i]))
        return categor_p,categor_r,categor_f1

    def calc(self,cfg,y_true,y_pred,best_epoch='1'):
        names = ["normal","abnormal"]

        keyword_df = pd.DataFrame()
        keyword_df['names']=names

        # 0->abnormal, 1->normal 
        y_true_normal = y_true[:,0]
        y_true_normal = 1 - y_true_normal

        #Method 1 argmax 
        y_pred_max = np.argmax(y_pred,axis=1)
        y_pred_normal = np.clip(y_pred_max, 0, 1)
        # y_pred_normal = 1 - y_pred_normal

        categor_p,categor_r,categor_f1 = self._cm_calc(y_true_normal, y_pred_normal)

        keyword_df['P_max_M1']=categor_p
        keyword_df['R_max_M1']=categor_r
        keyword_df['F1_max_M1']=categor_f1

        #Method 2 normal first
        y_pred_s = y_pred>0.5
        y_pred_s = y_pred_s*1
        y_pred_normal_m2 = y_pred_s[:,0]
        y_pred_normal_m2 = 1 - y_pred_normal_m2

        M2_categor_p,M2_categor_r,M2_categor_f1 = self._cm_calc(y_true_normal, y_pred_normal_m2)
        keyword_df['P_normFi_M2']=M2_categor_p
        keyword_df['R_normFi_M2']=M2_categor_r
        keyword_df['F1_normFi_M2']=M2_categor_f1

        #Method 3 abnormal first
        y_pred_abnormalraw = y_pred_s[:,1:]
        a=y_pred_abnormalraw
        y_pred_abnormalSum=np.dot(a, np.ones(a.shape[1]))
        y_pred_abnormal_m3 = np.clip(y_pred_abnormalSum, 0, 1)

        M3_categor_p,M3_categor_r,M3_categor_f1 = self._cm_calc(y_true_normal, y_pred_abnormal_m3)
        keyword_df['P_abnormFis_M3']=M3_categor_p
        keyword_df['R_abnormFis_M3']=M3_categor_r
        keyword_df['F1_abnormFis_M3']=M3_categor_f1
        out_path = cfg.output_dir+'/e{}_normal_abnormal_binary_eval_results.csv'.format(best_epoch)
        keyword_df.to_csv(out_path)

        cm = confusion_matrix(y_true_normal, y_pred_normal)
        save_path_values = cfg.output_dir+'/'+'e'+str(best_epoch)+'normal_abnormal_cm_max_M1.png'
        plot_confusion_matrix(cm,names,save_path_values,title='Confusion matrix values',cmap=None,normalize=False)

        cm = confusion_matrix(y_true_normal, y_pred_normal_m2)
        save_path_values = cfg.output_dir+'/'+'e'+str(best_epoch)+'normal_abnormal_cm_normF_M2.png'
        plot_confusion_matrix(cm,names,save_path_values,title='Confusion matrix values',cmap=None,normalize=False)

        cm = confusion_matrix(y_true_normal, y_pred_abnormal_m3)
        save_path_values = cfg.output_dir+'/'+'e'+str(best_epoch)+'normal_abnormal_cm_abnormF_M3.png'
        plot_confusion_matrix(cm,names,save_path_values,title='Confusion matrix values',cmap=None,normalize=False)
        print(out_path,' saved')

         


