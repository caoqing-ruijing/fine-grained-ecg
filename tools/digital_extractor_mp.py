import os
import re
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
from scipy import interpolate


class ECGLine():
    LEAD_NAMES = ['I', 'I I', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II']

    LEAD_COLORS = {'I': 'blueviolet', 'I I': 'mediumvioletred',
                   'III': 'navy', 'aVR': 'tomato', 'aVL': 'brown',
                   'aVF': 'darkmagenta', 'V1': 'limegreen', 'V2': 'indigo',
                   'V3': 'forestgreen', 'V4': 'magenta', 'V5': 'orange',
                   'V6': 'teal', 'II': 'orchid'}

    NORM_RELATIONS = {'I': 'I', 'aVR': 'I', 'V1': 'I', 'V4': 'I',
                      'I I': 'I I', 'aVL': 'I I', 'V2': 'I I', 'V5': 'I I',
                      'III': 'III', 'aVF': 'III', 'V3': 'III', 'V6': 'III',
                      'II': 'II'}

    def __init__(self, line, is_qrs=False, lead_name=None):
        """ Line for ECG data
        Args:
            line: [(x0,y0), (x1,y1), ...]
            is_qrs: whether the line belong QRS waves
            lead_name: one of `LEAD_NAMES`
        """
        self.line = line
        self.is_qrs = is_qrs
        self.lead_name = lead_name

    def __repr__(self):
        if self.lead_name is None:
            x = "ECG"
        else:
            x = self.lead_name
        return "%s's Line: " % x + str(self.line)

    def is_circular(self):
        """ whether the start point is same with the end point """
        x1, y1 = self.line[0]
        x2, y2 = self.line[-1]
        if x1 == x2 and y1 == y2:
            return True
        else:
            return False

    def is_connected(self, other_line):
        """ whether the end of this line is connected with the start of other line """
        x1, y1 = self.line[-1]
        x2, y2 = other_line.line[0]
        if (x1-x2) == 0 and (y1-y2) == 0:
            return True
        else:
            return False


def parse_svg(svg_path):
    # read the SVG file
    doc = minidom.parse(svg_path)
    # extract path and style
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    style_strings = [path.getAttribute('style') for path
                     in doc.getElementsByTagName('path')]
    doc.unlink()
    assert len(path_strings) == len(style_strings), "parse svg failed"
    style_set = list(set(style_strings))
    style_set.sort()
    return path_strings, style_strings, style_set


def extract_size(svg_path):
    """ extract size information from svg file """
    # original pdf size
    doc = minidom.parse(svg_path)
    tmp = doc._get_documentElement()
    w, h = [float(tmp.getAttribute(x).replace('pt', ''))
            for x in ['width', 'height']]
    doc.unlink()
    ori_size = (w, h)
    # svg's affine transform matrix
    with open(svg_path, 'r') as f:
        lines = f.readlines()
    transform_lines = filter(lambda x: "transform=" in x, lines)
    matrix_list = []
    for x in transform_lines:
        mat = re.search(r'transform="matrix\((.*)\)"', x)
        assert "matrix" in mat.group()
        matrix = mat.groups()[0]
        matrix_list.append(matrix)
    mat_counts = [(x, matrix_list.count(x)) for x in set(matrix_list)]
    mat_counts.sort(key=lambda x: x[1])
    most_mat = mat_counts[-1][0]  # assert the target matrix is at most
    most_mat = list(map(float, most_mat.split(',')))
    affine_mat = np.array(most_mat).reshape(3, 2).T.tolist()

    return {"affine_mat": affine_mat, "ori_size": ori_size}


def line2points(line):
    """ convert svg's line to points """
    x0 = line.start.real
    y0 = line.start.imag
    x1 = line.end.real
    y1 = line.end.imag
    points = [(x0, y0), (x1, y1)]
    return points


def is_long_straight_line(points, right_th=20, left_th=10):
    """ whether these points form a long vertical or horizontal line """
    right = (abs(points[0][1] - points[1][1]) > right_th) \
        and points[0][0] == points[1][0]
    left = (abs(points[0][0] - points[1][0]) > left_th) \
        and points[0][1] == points[1][1]
    return right or left


def extract_target_lines(svg_path):
    """ extract target lines from svg by rules """
    path_strings, style_strings, style_set = parse_svg(svg_path)

    draw_lines = []  # ECG curves
    start_path_list = []  # start path for each rows
    lead_break_lines = []  # lead split lines
    for i in range(0, len(path_strings)):
        # "stroke-width" is second-smallest (QRS waves)
        if style_strings[i] == style_set[1]:
            path = parse_path(path_strings[i])
            if len(path) >= 30:
                continue  # skip background
            for e in path:
                if type(e) is Line:
                    points = line2points(e)
                    if not is_long_straight_line(points):
                        draw_lines.append(ECGLine(points, True))
        # "stroke-width" is third-smallest (the rest waves)
        elif style_strings[i] == style_set[2]:
            path = parse_path(path_strings[i])
            if len(path) == 6:
                start_path_list.append(path)
                continue  # skip start signal
            for e in path:
                if type(e) is Line:
                    points = line2points(e)
                    if not is_long_straight_line(points):
                        draw_lines.append(ECGLine(points, False))
                    else:
                        lead_break_lines.append(points)
    assert len(lead_break_lines) == 9, "inferring lead break lines failed"
    return draw_lines, start_path_list, lead_break_lines


def start_path2line(path):
    """ infer the vertical line of the beginning of each row """
    start_points = []
    for e in path:
        if type(e) is Line:
            ps = line2points(e)
            start_points.extend(ps)
    x_max = max([x[0] for x in start_points])
    y_max = max([x[1] for x in start_points])
    y_min = min([x[1] for x in start_points])
    line = [(x_max, y_min), (x_max, y_max)]
    return line


def infer_coarse_lead_boxes(start_path_list, lead_break_lines):
    """ infer the coarse bounding boxes of leads """
    # start vertical lines
    start_lines = []
    if len(start_path_list) == 4:
        for p in start_path_list:
            line = start_path2line(p)
            start_lines.append(line)
    elif len(start_path_list) > 0:
        # complete the rest start lines
        lead_break_lines.sort(key=lambda x: x[1])
        lead_break_lines.sort(key=lambda x: x[0])
        lead_h = lead_break_lines[1][0][1]-lead_break_lines[0][0][1]
        start_line1 = start_path2line(start_path_list[0])
        p1, p2 = start_line1
        p1 = (p1[0], lead_break_lines[0][0][1])
        for i in range(4):
            p1_new = (p1[0], p1[1]+lead_h*i)
            p2_new = (p2[0], p2[1]+lead_h*i)
            start_lines.append([p1_new, p2_new])
    else:
        raise Exception("inferring start vertical lines failed")

    # start and mid vertical lines
    lead_lines = start_lines + lead_break_lines
    lead_lines.sort(key=lambda x: x[1])
    lead_lines.sort(key=lambda x: x[0])

    # bottom row's vertical lead line
    bottom_lead_line = lead_lines.pop(3)
    pt1, pt2 = bottom_lead_line
    h = pt2[1]-pt1[1]
    pt1 = (pt1[0], pt1[1])
    pt2 = (pt2[0], pt2[1]+h)
    bottom_lead_line = [pt1, pt2]

    # top 3 row's final vertical lines
    lead_width = lead_lines[3][0][0]-lead_lines[0][0][0]
    end_lines = []
    for pt1, pt2 in lead_lines[-3:]:
        pt1 = (pt1[0]+lead_width, pt1[1])
        pt2 = (pt2[0]+lead_width, pt2[1])
        end_lines.append([pt1, pt2])
    # bottom row's final vertical line
    pt1, pt2 = bottom_lead_line
    pt1 = (pt1[0]+lead_width*4, pt1[1])
    pt2 = (pt2[0]+lead_width*4, pt2[1])
    end_lines.append([pt1, pt2])

    # lead boxes
    lead_names = ECGLine.LEAD_NAMES
    lead_dict = {}
    for i in range(len(lead_names)):
        if i < 9:
            box = [lead_lines[i][0], lead_lines[i+3][1]]
        elif i == 12:
            box = [bottom_lead_line[0], end_lines[-1][1]]
        else:
            box = [lead_lines[i][0], end_lines[i-9][1]]
        lead_dict[lead_names[i]] = box

    return lead_dict


def get_points_bbox(points, epsilon=0.0):
    """ get the bounding box according to the points
    Args:
        points: [(x_0,y_0), (x_1,y_1), ...]
        epsilon: add epsilon when (x_min==x_max) or (y_min==y_max)
    Return:
        bbox: [(x_min,y_min), x_max,y_max)]
    """
    points = np.array(points)
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    if x_min == x_max:
        x_max += epsilon
    if y_min == y_max:
        y_max += epsilon
    bbox = [(x_min, y_min), (x_max, y_max)]
    return bbox


def box_contains_point(box, points):
    """ whether the box contains the point """
    box = np.array(box)
    points = np.array(points)
    xs, ys = points[:, 0], points[:, 1]
    c1 = np.less_equal(box[0][0], xs)
    c2 = np.less_equal(xs, box[1][0])
    c3 = np.less_equal(box[0][1], ys)
    c4 = np.less_equal(ys, box[1][1])
    return c1 & c2 & c3 & c4


def process_lead_curve(draw_lines, coarse_box_dict, min_length=1100):
    """ infer and inplace `lead_name` for all elements in `draw_lines` """
    # find break ranges
    break_list = []  # contains the lead ranges
    extra_inds = []
    line_pre = draw_lines[0]
    i_pre = 0
    for i, line_now in enumerate(draw_lines[1:]):
        i += 1  # recorrect index
        if not line_pre.is_connected(line_now):
            x_now, x_pre = line_now.line[0][0], line_pre.line[1][0]
            len_x = i-i_pre
            # check whether `(i_pre, i)`` is a break range
            if line_now.is_circular() or (x_now < x_pre) or (1210<len_x<1250) or (4000<len_x<4800):
                if (line_now.is_circular() and len_x>500) or (len_x>min_length): # ignore incomplete break range
                    # `line_now` is the start of one lead curve
                    break_list.append((i_pre, i))
            else:
                extra_inds.append(i)
            i_pre = i
        line_pre = line_now
    # each elem in `brea_list` means
    assert len(break_list) <= 13, "break lines failed"
    if (len(break_list)<13) and ((len(draw_lines)-i_pre)>2000): # ignore too short "II" lead curves
        break_list.append((i_pre, len(draw_lines)))
    if len(extra_inds) > 0:
        # remove break which is out of the svg's boundary
        break_list_new = []
        for b in break_list:
            flag = True
            for e in extra_inds:
                if b[0] < e <= b[1]:
                    flag = False
            if flag:
                break_list_new.append(b)
        break_list = break_list_new

    # infer lead name
    lead_names = ECGLine.LEAD_NAMES
    break_dict = {}
    for b in break_list:
        lead_path = draw_lines[b[0]:b[1]]
        lead_points = []
        for x in lead_path:
            # ignore QRS since its wild fluctuations
            if not x.is_qrs:
                lead_points.extend(x.line)
        # the related lead contains the most points
        contain_counts = [box_contains_point(
            coarse_box_dict[n], lead_points).sum() for n in lead_names]
        name = lead_names[np.argmax(contain_counts)]
        max_count = max(contain_counts)
        if name in break_dict:
            # assign lead name to `break` which contains the most points
            if break_dict[name][1] < max_count:
                break_dict[name] = [b, max_count]
        else:
            break_dict[name] = [b, max_count]
    # inplace lead name
    for name in break_dict:
        b = break_dict[name][0]
        lead_path = draw_lines[b[0]:b[1]]
        for x in lead_path:
            x.lead_name = name
    return draw_lines


def build_digital_dict(draw_lines):
    """ build a dict for digital data.  
    Return:
        {
            "I": {"path": [(x_0,y_0), (x_1,y_1), ... , (x_n,y_n)],
                  "qrs": [(i,j), (k,l), ...]}, 
            "I I": {"path": ...,
                    "qrs": ...}, 
            ...
        }
    """
    digital_dict = {}
    for n in ECGLine.LEAD_NAMES:
        digital_dict[n] = {"path": [], "qrs": [], "tmp": []}
    # raw lines
    for x in draw_lines:
        if x.lead_name is not None:
            digital_dict[x.lead_name]["tmp"].append(x)
    # ECG path for each lead
    for key in digital_dict.keys():
        tmp = digital_dict[key]["tmp"]
        if len(tmp) > 1:
            if not tmp[0].is_circular():
                lead_curve = list(tmp[0].line)
            else:
                lead_curve = [tmp[0].line[1]]
            line_pre = tmp[0]
            for line_now in tmp[1:]:
                assert line_pre.is_connected(line_now), "the lead is not consecutive"
                lead_curve.append(line_now.line[1])
                line_pre = line_now
            digital_dict[key]["path"] = lead_curve.copy()
    # QRS waves for each lead
    for key in digital_dict.keys():
        tmp = digital_dict[key]["tmp"]
        path = digital_dict[key]["path"]
        i = 0
        while i < len(tmp):
            while i < len(tmp) and (not tmp[i].is_qrs):
                i += 1
            if i == (len(tmp)):
                break
            j = i+1
            while j < len(tmp) and tmp[j].is_qrs:
                j += 1
            if j > (i+2):  # ignore too short range
                start = path.index(tmp[i].line[0])
                end = path.index(tmp[j-1].line[1])
                digital_dict[key]["qrs"].append((start, end))
            i = j+1
    # pop raw lines
    for key in digital_dict.keys():
        digital_dict[key].pop("tmp")
    return digital_dict


def affine_transform(path, affine_mat):
    """ apply affine transform to `path` 
    Args: 
        path: [(x_0,y_0), (x_1,y_1), ...]
        affine_mat: [[a,b,c], [d,e,f]]
    """
    path_np = np.concatenate([np.array(path), np.ones((len(path), 1))], axis=1)
    path_np = path_np[:, :, None]
    affine_mat_np = np.array(affine_mat)[None, :, :]
    path_np = np.matmul(affine_mat_np, path_np).squeeze()
    path_new = [tuple(x) for x in path_np.tolist()]
    return path_new


def generate_save_path(ori_path, save_root, suffix='.png'):
    """ generate save path with the same filename"""
    base = os.path.basename(ori_path)
    tmp = '.'+base.split('.')[-1]
    save_path = os.path.join(save_root, base.replace(tmp, suffix))
    return save_path


def tuple_inside(x):
    """ [[a,b],[c,d]] -> [(a,b),(c,d)]"""
    return list(map(tuple, x))


def draw_ecg_with_coarse_qrs(digital_dict, only_path=False, lead_colors=None,
                             qrs_width=0, rest_width=0, font_size=12, im=None):
    """ visualize extraction result """
    if lead_colors is None:
        lead_colors = ECGLine.LEAD_COLORS
    if im is None:
        draw_size = digital_dict["info"]["size"]
        draw_size = tuple(np.ceil(draw_size).astype(int))
        im = Image.new('RGB', draw_size, (255, 255, 255))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("Arial Unicode", font_size)
    coarse_bbox = digital_dict["info"]["coarse_bbox"]
    for key in coarse_bbox.keys():
        c = lead_colors[key]
        path = digital_dict[key]["path"]
        # annotation boxes
        if not only_path:
            # lead box and text
            lead_box = coarse_bbox[key]
            draw.rectangle(tuple_inside(lead_box), outline='gray')
            draw.text(tuple(lead_box[0]), key, fill=c, font=font)
            # qrs bbox
            for qrs in digital_dict[key]["qrs"]:
                bbox = get_points_bbox(path[qrs[0]:qrs[1]+1])
                draw.rectangle(tuple_inside(bbox), outline='silver')
        # different widths for qrs and the rest waves
        post = np.array(path)
        qrs_list = []
        rest_list = []
        ind = 0
        for q in digital_dict[key]["qrs"]:
            pre, mid, post = np.split(post, [q[0]-ind, q[1]+1-ind])
            if len(post) > 0:
                tmp = mid.tolist()+[post[0].tolist()]
            else:
                tmp = mid.tolist()
            qrs_list.append(tmp)
            if len(mid) > 0:
                tmp = pre.tolist()+[mid[0].tolist()]
            else:
                tmp = pre.tolist()
            ind += len(pre)+len(mid)
            rest_list.append(tmp)
        if len(post) > 0:
            rest_list.append(post.tolist())
        for qrs in qrs_list:
            draw.line(tuple_inside(qrs), fill=c, width=qrs_width)
        for rest in rest_list:
            draw.line(tuple_inside(rest), fill=c, width=rest_width)
    return im


def extract_digital_from_svg(svg_path, save_root=None, save_json=True, save_vis=False):
    """ extract digital result form svg file  
    Args:
        svg_path: the path of the svg file
        save_root: the root for saving result
        save_json: whether save result as json file
        save_vis: whether save the visualization of the result  
    Return:
        {
            "info": {
                "size": (w, h), 
                "coarse_bbox": {
                        "I":[(x_0,y_0), (x_1,y_1)], 
                        "I I": [...], 
                        ...
                                } # coarse bounding box for each lead
                    }
            "I": {
                "path": [(x_0,y_0), (x_1,y_1), ... , (x_n,y_n)], # lead curve
                "qrs": [(i,j), (k,l), ...] # the start and end indexes of QRS wave
                 }, 
            "I I": {
                "path": ...,
                "qrs": ...
                }, 
            ...
        }
    """
    try:
        # extract digital
        size_dict = extract_size(svg_path)
        ori_size = size_dict["ori_size"]
        affine_mat = size_dict["affine_mat"]
        draw_lines, start_path_list, lead_break_lines = extract_target_lines(svg_path)
        coarse_box_dict = infer_coarse_lead_boxes(start_path_list, lead_break_lines)
        process_lead_curve(draw_lines, coarse_box_dict)
        result = build_digital_dict(draw_lines)
        for key in result.keys():
            path = result[key]["path"]
            if len(path) > 0:
                # resize to original size
                result[key]["path"] = affine_transform(path, affine_mat)
            coarse_box_dict[key] = affine_transform(coarse_box_dict[key], affine_mat)
        result["info"] = {"size": ori_size, "coarse_bbox": coarse_box_dict}
        # save reuslt
        if save_json:
            save_path = generate_save_path(svg_path, save_root, '.json')
            with open(save_path, "w") as f:
                json.dump(result, f)
        if save_vis:
            img = draw_ecg_with_coarse_qrs(result)
            save_path = generate_save_path(svg_path, save_root, '.png')
            img.save(save_path)
        return result
    except Exception as e:
        print("error in", svg_path, ':',  e)
        if save_json:
            # save the failed case as a text file
            save_path = generate_save_path(svg_path, save_root, '.txt')
            with open(save_path, "w") as f:
                f.write(str(e))


def get_filelist(root, suffix=".pdf"):
    """ recursively obtain file paths which contain `suffix` from `root` """
    Filelist = []
    for home, _, files in os.walk(root):
        for filename in files:
            if filename.endswith(suffix):
                Filelist.append(os.path.join(home, filename))
    return Filelist


def pdf2svg_fn(pdf_path, save_root):
    """ convert pdf to svg file by command line """
    svg_path = generate_save_path(pdf_path, save_root, '.svg')
    os.system("pdf2svg %s %s" % ((pdf_path, svg_path)))


def normalize_path(result):
    """ both x and y start at zero for each lead """
    path_dict = {}
    for (key, value) in ECGLine.NORM_RELATIONS.items():
        path1 = result[key]["path"]
        path2 = result[value]["path"]
        if len(path1) > 0:
            x = path1[0][0]
            if len(path2) > 0:
                y = path2[0][1]
            else:
                y = path1[0][1]
            norm_path = np.array(path1)-np.array([(x, y)])
        else:
            norm_path = []
        path_dict[key] = norm_path
    return path_dict


def point2digital(json_path, save_root, sample_num=1250, nan_value=-1):
    """ convert ECG points to digital values with fixed length 
    Return:
        {
            "info": {
                "size": (w, h), 
                "coarse_bbox": {
                        "I":[(x_0,y_0), (x_1,y_1)], 
                        "I I": [...], 
                        ...
                                } # coarse bounding box for each lead
                    }
            "I": {
                "value": [y_0, y_1, ... , y_n], # lead values (its length is `sample_num`)
                "qrs_align": [(i,j), (k,l), ...] # the start and end indexes of QRS wave
                 }, 
            "I I": {
                "value": ...,
                "qrs_align": ...
                }, 
            ...
        }
    """
    with open(json_path, 'r') as f:
        result = json.load(f)
    norm_paths = normalize_path(result)
    for key, path in norm_paths.items():
        if key == "II":
            sample_num *= 4
        if len(path) > 0:
            xs_even = np.linspace(0, path[:, 0].max(), sample_num)
            inter = interpolate.interp1d(
                path[:, 0], -path[:, 1], kind='linear')
            ys_inter = inter(xs_even).tolist()
            qrs_align = np.array(result[key]["qrs"]) / \
                len(result[key]["path"]) * sample_num
            qrs_align = np.round(qrs_align).astype('int').tolist()
        else:
            ys_inter = [nan_value]*sample_num
            qrs_align = nan_value
        result[key] = {"value": ys_inter, "qrs_align": qrs_align}
    save_path = generate_save_path(json_path, save_root, '.json')
    with open(save_path, "w") as f:
        json.dump(result, f)


if __name__ == '__main__':
    from tqdm import tqdm
    from argparse import ArgumentParser
    import multiprocessing
    from functools import partial

    parser = ArgumentParser(
        description="Convert ECG svg to digital json (or 'pdf to svg' or 'refine') in mulitprocess")
    parser.add_argument("--root", type=str,
                        help="the root of input files (.txt)")
    parser.add_argument("--output", type=str,
                        help="the directory for saving outputs")
    parser.add_argument("--pdf2svg", action="store_true",
                        help="convert pdf to svg (skip svg to json)")
    parser.add_argument("--refine", action="store_true",
                        help="convert points to digital and fixs length")
    parser.add_argument("--visualize", action="store_true",
                        help="visualize extraction result")
    parser.add_argument("--cpu", type=int, default=4,
                        help="the number of cpu cores for working")

    args = parser.parse_args()
    assert (args.pdf2svg+args.refine+args.visualize) <= 1, "confict in pdf2svg, refine, visualize"
    save_root = args.output
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if args.pdf2svg:
        # convert pdf to svg
        path_list = get_filelist(args.root, ".pdf")
        fn = partial(pdf2svg_fn, save_root=save_root)
    elif args.refine:
        # convert points to digital and fixs length
        path_list = get_filelist(args.root, ".json")
        fn = partial(point2digital, save_root=save_root)
    else:
        # extract json file from svg
        path_list = get_filelist(args.root, ".svg")
        fn = partial(extract_digital_from_svg,
                     save_root=save_root,
                     save_json=True,
                     save_vis=args.visualize)
    if args.cpu > 1:
        # multi-process
        pool = multiprocessing.Pool(processes=args.cpu)
        with tqdm(total=len(path_list)) as progress_bar:
            for _ in pool.imap_unordered(fn, path_list):
                progress_bar.update(1)
    else:
        # single process
        for path in tqdm(path_list):
            fn(path)
    print('Done')
