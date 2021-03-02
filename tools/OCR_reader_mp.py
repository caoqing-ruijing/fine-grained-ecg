# -*- coding: utf-8 -*-

import sys
import importlib
importlib.reload(sys)

from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import *
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage, LTChar
import pdf2image

import os,sys,io
from PIL import Image
import  pandas  as pd
import numpy as np

'''
解析pdf文件，获取文件中包含的各种对象
'''

LEAD_NAMES = ['I','I I','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','II']
EXTRA_LEAD_NAMES = ['V3R', 'V4R', 'V5R', 'V7', 'V8', 'V9']
LEAD_NAMES.extend(EXTRA_LEAD_NAMES)

# 解析pdf文件函数
def parse(pdf_path):
    fp = open(pdf_path, 'rb')  # 以二进制读模式打开
    # 用文件对象来创建一个pdf文档分析器
    parser = PDFParser(fp)
    # 创建一个PDF文档
    doc = PDFDocument()
    # 连接分析器 与文档对象
    parser.set_document(doc)
    doc.set_parser(parser)

    # 提供初始化密码
    # 如果没有密码 就创建一个空的字符串
    doc.initialize()

    rs = {}
    rs[u'心电图诊断'] = ''
    rs[u'性别'] = ''
    rs[u'年龄'] = ''
    rs[u'心率'] = ''
    rs[u'P-R'] = ''
    rs[u'QT/QTc'] = ''
    rs[u'QRS'] = ''
    rs[u'QRS电轴'] = ''

    for n in LEAD_NAMES:
        rs[n] = ''
    # 检测文档是否提供txt转换，不提供就忽略
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        # 创建PDf 资源管理器 来管理共享资源
        rsrcmgr = PDFResourceManager()
        # 创建一个PDF设备对象
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # 创建一个PDF解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # 用来计数页面，图片，曲线，figure，水平文本框等对象的数量
        num_page, num_image, num_curve, num_figure, num_TextBoxHorizontal = 0, 0, 0, 0, 0
        pic_num = 0
        
        # 循环遍历列表，每次处理一个page的内容
        for page in doc.get_pages(): # doc.get_pages() 获取page列表
            num_page += 1  # 页面增一
            interpreter.process_page(page)
            # 接受该页面的LTPage对象
            layout = device.get_result()
            for x in layout:
                if isinstance(x,LTImage):  # 图片对象
                    num_image += 1
                    print(x.get_)
                if isinstance(x,LTCurve):  # 曲线对象
                    num_curve += 1
                    
                if isinstance(x,LTFigure):  # figure对象
                    num_figure += 1
                    
                    # for each_x in x:
                    #     if each_x.stream:
                    #         buffer = io.BytesIO(each_x.stream.get_data())
        
                    #         with open("figure_data_{}.jpg".format(pic_num), 'wb') as fp :
                    #             fp.write(buffer.read())
                                
                    #         pic_num += 1
        
                if isinstance(x, LTTextBoxHorizontal):  # 获取文本内容
                    num_TextBoxHorizontal += 1  # 水平文本框对象增一
                    results = x.get_text()

                    if results.strip() in LEAD_NAMES:
                        rs[results.strip()] = x.bbox
                    
                    if u'心电图诊断' in results:
                        results = results #.replace('\n', '').replace(' ', '').replace(u' ', '')
                        items = results.split(':')
                        name = items[0].strip()
                        
                        if len(items)>1 and len(name.strip())>0:
                            rs[name] = items[1]
                    elif u'检查时间' in results:
                        results = results.replace('\n', '')
                        items = results.split(' ')
                        
                        front = ''.join(items[0:-1]).replace(' ', '')
                        end = items[-1].replace(' ', '')
                        
                        front_items = front.split(':')
                        name = front_items[0].strip()
                        
                        #if name not in rs.keys() and len(name.strip())>0:
                        rs[name] = front_items[1] + ' ' + end
                        
                        
                    else:
                        results = results.split('\n')
                        for each_result in results:
                            each_result = each_result.replace(' ', '').replace(u' ', '')
                            items = each_result.split(':')
                            name = items[0].strip()
                            
                            if len(items)>1 and len(name.strip())>0:
                                rs[name] = items[1]
                        
    return rs


def get_filelist(dir):
    
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.pdf'):
                Filelist.append(os.path.join(home, filename))
    return Filelist


def pdf_parse_single(index_file):
    index, file_path = index_file

    ECG_ID = file_path.split('/')[-1].replace('.pdf', '')
    nan_value = np.nan # -1
    Gender = Age = HR = PR = QT = QTc = QRS = QRSE = nan_value
    try:
        pdf_size = pdf2image.pdfinfo_from_path(file_path)['Page size']
    except:
        pdf_size ='-1'
    try:
        rs = parse(file_path)

        if '男' in str(rs[u'性别']):
            Gender = 1
        if '女' in str(rs[u'性别']):
            Gender = 0
        
        if '不详' in str(rs[u'年龄']):
            Age = nan_value
        else:
            Age = str(rs[u'年龄']).replace(u'岁', '')
        
        HR = str(rs[u'心率']).replace('bpm', '')  
        PR = str(rs['P-R']).replace('ms', '').replace('/', '')
        items = str(rs['QT/QTc']).replace('ms', '').split('/')
        if len(items) == 2:
            QT = items[0]
            QTc = items[1]
            
        QRS = str(rs['QRS']).replace('ms', '')
        QRSE = str(rs['QRS电轴']).replace('ms', '').replace('°', '').replace('+', '').replace('/', '').strip()
        items = rs['心电图诊断'].split('\n')
        items = [x.strip().split('、')[-1] for x in items if len(x.strip())>0]
        lead_list = []
        for lead_name in LEAD_NAMES:
            bbox_str = ' '.join(map(str, rs[lead_name]))
            lead_list.append(bbox_str)

    except:
        items = ["-1"]
        lead_list = ["-1"] * len(LEAD_NAMES)
    row = [ECG_ID, Gender, Age, HR, PR, QT, QTc, QRS, QRSE, '###'.join(items)]
    row.extend(lead_list)
    row.append(pdf_size)
    return index, row, items
    
# 输入转为一个带index的生成器
def generate_args(data):
    for i in tqdm(range(len(data))):
        yield (i, data[i])

if __name__ == '__main__':
    import multiprocessing as mp
    from tqdm import tqdm
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="OCR Reader in mulitprocess")
    parser.add_argument("--root", type=str,
                        help="the root of input pdfs")
    parser.add_argument("--output", type=str, default="./",
                        help="the directory for saving outputs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="the number of cpu cores for working")
    args = parser.parse_args()

    columns_list = ["ECG_ID", "Gender", "Age", "HR", "PR", "QT", "QTc", "QRS", "QRSE", "Label"]
    columns_list.extend(LEAD_NAMES)
    columns_list.append("PDF_size")
    df_new = pd.DataFrame(columns = columns_list)
    label_count = {}
    
    path = args.root
    statistical_rs = os.path.join(args.output, "statistical_rs.txt")
    profile_rs = os.path.join(args.output, "save_data_profile_v3.csv")
    num_workers = args.num_workers
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    Filelist = get_filelist(path)
    print(len( Filelist))
    with mp.Pool(num_workers) as pool:
        imap_it = pool.imap_unordered(pdf_parse_single, generate_args(Filelist))
        for index, row, items in imap_it:
            df_new.loc[index] = row # [ECG_ID, Gender, Age, HR, PR, QT, QTc, QRS, QRSE, '###'.join(items)]
            for each in items:
                if each in label_count.keys():
                    label_count[each] += 1
                else:
                    label_count[each] = 1
        
    df_new.to_csv(profile_rs,index = False)
    with open(statistical_rs, 'w') as f:
        for key in label_count.keys():
            f.write(key + '\t' + str(label_count[key]) + '\n')
    print('Done')
    
    