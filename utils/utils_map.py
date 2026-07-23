
import glob
import json
import math
import operator
import os
import shutil
import sys
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    pass
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
def error(msg):
    """
    throw error and exit
    """    
    print(msg)
    sys.exit(0)


def file_lines_to_list(path):
    """
    Convert the lines of a file to a list
    """
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def is_float_between_0_and_1(value):
    """
     check if the number is a float between 0.0 and 1.0
    """
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def draw_text_in_image(img, text, pos, color, line_width):
    """
    Draws text in image
    """
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            color,
            lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    """
    Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
    Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - orange -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.manager.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()




def preprocess_gt(gt_path, class_names):
    """Convert ground-truth txt files into COCO-style dictionaries."""
    image_ids   = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i, image_id in enumerate(image_ids):
        lines_list      = file_lines_to_list(os.path.join(gt_path, image_id))
        boxes_per_image = []
        image           = {}
        image_id        = os.path.splitext(image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width']     = 1
        image['height']    = 1
        image['id']        = str(image_id)

        for line in lines_list:
            difficult = 0 
            if "difficult" in line:
                line_split  = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name  = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name  = class_name[:-1]
                difficult = 1
            else:
                line_split  = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name  = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]
            
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            if class_name not in class_names:
                continue
            cls_id  = class_names.index(class_name) + 1
            bbox    = [left, top, right - left, bottom - top, difficult, str(image_id), cls_id, (right - left) * (bottom - top) - 10.0]
            boxes_per_image.append(bbox)
        images.append(image)
        bboxes.extend(boxes_per_image)
    results['images']        = images

    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory']   = cls
        category['name']            = cls
        category['id']              = i + 1
        categories.append(category)
    results['categories']   = categories

    annotations = []
    for i, box in enumerate(bboxes):
        annotation = {}
        annotation['area']        = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id']    = box[-3]
        annotation['iscrowd']     = box[-4]
        annotation['bbox']        = box[:4]
        annotation['id']          = i
        annotations.append(annotation)
    results['annotations'] = annotations
    return results


def preprocess_dr(dr_path, class_names):
    """Convert detection-result txt files into COCO-style detections."""
    image_ids = os.listdir(dr_path)
    results = []
    for image_id in image_ids:
        lines_list      = file_lines_to_list(os.path.join(dr_path, image_id))
        image_id        = os.path.splitext(image_id)[0]
        for line in lines_list:
            line_split  = line.split()
            confidence, left, top, right, bottom = line_split[-5:]
            class_name  = ""
            for name in line_split[:-5]:
                class_name += name + " "
            class_name  = class_name[:-1]
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            result                  = {}
            result["image_id"]      = str(image_id)
            if class_name not in class_names:
                continue
            result["category_id"]   = class_names.index(class_name) + 1
            result["bbox"]          = [left, top, right - left, bottom - top]
            result["score"]         = float(confidence)
            results.append(result)
    return results


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        Calculate log-average miss rate by averaging miss rates at 9 evenly
        spaced FPPI points between 10e-2 and 10e0 in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def voc_ap(rec, prec):
    """
    Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
            precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path='./map_out', return_class_metrics=False):
    """Compute VOC-style mAP and optional per-class F1/Recall/Precision."""

    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')

    show_animation = False
    if os.path.exists(IMG_PATH):
        for _, _, files in os.walk(IMG_PATH):
            if files:
                show_animation = True
                break

    if os.path.exists(TEMP_FILES_PATH):
        shutil.rmtree(TEMP_FILES_PATH)
    os.makedirs(TEMP_FILES_PATH, exist_ok=True)

    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    os.makedirs(RESULTS_FILES_PATH, exist_ok=True)

    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            pass
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'AP'), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'F1'), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'Recall'), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'Precision'), exist_ok=True)
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'images', 'detections_one_by_one'), exist_ok=True)

    ground_truth_files_list = sorted(glob.glob(os.path.join(GT_PATH, '*.txt')))
    if len(ground_truth_files_list) == 0:
        error('Error: No ground-truth files found!')

    gt_counter_per_class = {}
    counter_images_per_class = {}

    # Build per-image GT json files.
    for txt_file in ground_truth_files_list:
        file_id = os.path.splitext(os.path.basename(txt_file))[0]
        det_file = os.path.join(DR_PATH, file_id + '.txt')
        if not os.path.exists(det_file):
            error('Error. File not found: {}\n'.format(det_file))

        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        already_seen_classes = []

        for line in lines_list:
            line_split = line.split()
            if len(line_split) < 5:
                continue

            difficult_flag = False
            if line_split[-1] == 'difficult':
                difficult_flag = True
                line_split = line_split[:-1]

            left, top, right, bottom = line_split[-4:]
            class_name = ' '.join(line_split[:-4]).strip()
            if class_name == '':
                continue

            bbox = left + ' ' + top + ' ' + right + ' ' + bottom
            if difficult_flag:
                bounding_boxes.append({'class_name': class_name, 'bbox': bbox, 'used': False, 'difficult': True})
            else:
                bounding_boxes.append({'class_name': class_name, 'bbox': bbox, 'used': False})
                gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1
                if class_name not in already_seen_classes:
                    counter_images_per_class[class_name] = counter_images_per_class.get(class_name, 0) + 1
                    already_seen_classes.append(class_name)

        with open(os.path.join(TEMP_FILES_PATH, file_id + '_ground_truth.json'), 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = sorted(list(gt_counter_per_class.keys()))
    n_classes = len(gt_classes)

    dr_files_list = sorted(glob.glob(os.path.join(DR_PATH, '*.txt')))

    # Build per-class detection json files.
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []

        for txt_file in dr_files_list:
            file_id = os.path.splitext(os.path.basename(txt_file))[0]
            gt_file = os.path.join(GT_PATH, file_id + '.txt')
            if class_index == 0 and not os.path.exists(gt_file):
                error('Error. File not found: {}\n'.format(gt_file))

            lines = file_lines_to_list(txt_file)
            for line in lines:
                line_split = line.split()
                if len(line_split) < 6:
                    continue

                bottom = line_split[-1]
                right = line_split[-2]
                top = line_split[-3]
                left = line_split[-4]
                confidence = line_split[-5]
                tmp_class_name = ' '.join(line_split[:-5]).strip()

                if tmp_class_name == class_name:
                    bbox = left + ' ' + top + ' ' + right + ' ' + bottom
                    bounding_boxes.append({'confidence': confidence, 'file_id': file_id, 'bbox': bbox})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(os.path.join(TEMP_FILES_PATH, class_name + '_dr.json'), 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    class_metrics = {}

    with open(os.path.join(RESULTS_FILES_PATH, 'results.txt'), 'w') as results_file:
        results_file.write('# AP and precision/recall per class\n')
        count_true_positives = {}

        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_data = json.load(open(os.path.join(TEMP_FILES_PATH, class_name + '_dr.json')))

            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            score_threhold_idx = 0

            for idx, detection in enumerate(dr_data):
                file_id = detection['file_id']
                score[idx] = float(detection['confidence'])
                if score[idx] >= score_threhold:
                    score_threhold_idx = idx

                gt_data = json.load(open(os.path.join(TEMP_FILES_PATH, file_id + '_ground_truth.json')))

                ovmax = -1
                gt_match = None
                bb = [float(x) for x in detection['bbox'].split()]

                for obj in gt_data:
                    if obj['class_name'] != class_name:
                        continue
                    bbgt = [float(x) for x in obj['bbox'].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

                if ovmax >= MINOVERLAP and gt_match is not None:
                    if 'difficult' not in gt_match:
                        if not bool(gt_match['used']):
                            tp[idx] = 1
                            gt_match['used'] = True
                            count_true_positives[class_name] += 1
                            with open(os.path.join(TEMP_FILES_PATH, file_id + '_ground_truth.json'), 'w') as f:
                                f.write(json.dumps(gt_data))
                        else:
                            fp[idx] = 1
                else:
                    fp[idx] = 1

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val

            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, _ in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            for idx, _ in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1, (np.array(prec) + np.array(rec)))

            sum_AP += ap
            text = '{0:.2f}%'.format(ap * 100) + ' = ' + class_name + ' AP '

            if len(prec) > 0:
                f1_value = float(F1[score_threhold_idx])
                recall_value = float(rec[score_threhold_idx])
                precision_value = float(prec[score_threhold_idx])
                F1_text = '{0:.2f}%'.format(f1_value * 100) + ' = ' + class_name + ' F1 '
                Recall_text = '{0:.2f}%'.format(recall_value * 100) + ' = ' + class_name + ' Recall '
                Precision_text = '{0:.2f}%'.format(precision_value * 100) + ' = ' + class_name + ' Precision '
            else:
                f1_value = 0.0
                recall_value = 0.0
                precision_value = 0.0
                F1_text = '0.00% = ' + class_name + ' F1 '
                Recall_text = '0.00% = ' + class_name + ' Recall '
                Precision_text = '0.00% = ' + class_name + ' Precision '

            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + '\n Precision: ' + str(rounded_prec) + '\n Recall :' + str(rounded_rec) + '\n\n')

            print(
                text + '\t||\tscore_threhold=' + str(score_threhold)
                + ' : F1=' + '{0:.2f}%'.format(f1_value * 100)
                + ' ; Recall=' + '{0:.2f}%'.format(recall_value * 100)
                + ' ; Precision=' + '{0:.2f}%'.format(precision_value * 100)
            )

            class_metrics[class_name] = {
                'f1': f1_value,
                'recall': recall_value,
                'precision': precision_value,
            }
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class.get(class_name, 0)
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            if draw_plot:
                plt.plot(rec, prec, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                fig = plt.gcf()
                fig.canvas.manager.set_window_title('AP ' + class_name)
                plt.title('class: ' + text)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, 'AP', class_name + '.png'))
                plt.cla()

                plt.plot(score, F1, '-', color='orangered')
                plt.title('class: ' + F1_text + '\nscore_threhold=' + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('F1')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, 'F1', class_name + '.png'))
                plt.cla()

                plt.plot(score, rec, '-H', color='gold')
                plt.title('class: ' + Recall_text + '\nscore_threhold=' + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('Recall')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, 'Recall', class_name + '.png'))
                plt.cla()

                plt.plot(score, prec, '-s', color='palevioletred')
                plt.title('class: ' + Precision_text + '\nscore_threhold=' + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, 'Precision', class_name + '.png'))
                plt.cla()

        if n_classes == 0:
            print('No classes found in ground-truth. Please check labels and classes_path.')
            if return_class_metrics:
                return 0, {}, {}
            return 0, {}

        results_file.write('\n# mAP of all classes\n')
        mAP = sum_AP / n_classes
        text = 'mAP = {0:.2f}%'.format(mAP * 100)
        results_file.write(text + '\n')
        print(text)

    shutil.rmtree(TEMP_FILES_PATH)

    det_counter_per_class = {}
    for txt_file in dr_files_list:
        for line in file_lines_to_list(txt_file):
            class_name = line.split()[0]
            det_counter_per_class[class_name] = det_counter_per_class.get(class_name, 0) + 1
    dr_classes = list(det_counter_per_class.keys())

    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    with open(os.path.join(RESULTS_FILES_PATH, 'results.txt'), 'a') as results_file:
        results_file.write('\n# Number of ground-truth objects per class\n')
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ': ' + str(gt_counter_per_class[class_name]) + '\n')

        results_file.write('\n# Number of detected objects per class\n')
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ': ' + str(n_det)
            text += ' (tp:' + str(count_true_positives[class_name])
            text += ', fp:' + str(n_det - count_true_positives[class_name]) + ')\n'
            results_file.write(text)

    if draw_plot:
        window_title = 'ground-truth-info'
        plot_title = 'ground-truth\n'
        plot_title += '(' + str(len(ground_truth_files_list)) + ' files and ' + str(n_classes) + ' classes)'
        x_label = 'Number of objects per class'
        output_path = os.path.join(RESULTS_FILES_PATH, 'ground-truth-info.png')
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            False,
            'forestgreen',
            '',
        )

        window_title = 'lamr'
        plot_title = 'log-average miss rate'
        x_label = 'log-average miss rate'
        output_path = os.path.join(RESULTS_FILES_PATH, 'lamr.png')
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            False,
            'royalblue',
            '',
        )

        window_title = 'mAP'
        plot_title = 'mAP = {0:.2f}%'.format(mAP * 100)
        x_label = 'Average Precision'
        output_path = os.path.join(RESULTS_FILES_PATH, 'mAP.png')
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            True,
            'royalblue',
            '',
        )

    if return_class_metrics:
        return mAP, ap_dictionary, class_metrics
    return mAP, ap_dictionary


def get_coco_map(class_names, path):
    """Run COCO mAP evaluation on prepared detection and ground-truth files."""
    GT_PATH     = os.path.join(path,  'ground-truth')
    DR_PATH     = os.path.join(path, 'detection-results')
    COCO_PATH   = os.path.join(path, 'coco_eval')

    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH)

    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

    with open(GT_JSON_PATH, "w") as f:
        results_gt  = preprocess_gt(GT_PATH, class_names)
        json.dump(results_gt, f, indent=4)

    with open(DR_JSON_PATH, "w") as f:
        results_dr  = preprocess_dr(DR_PATH, class_names)
        json.dump(results_dr, f, indent=4)
        if len(results_dr) == 0:
            print("No detections found.")
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cocoGt      = COCO(GT_JSON_PATH)
    cocoDt      = cocoGt.loadRes(DR_JSON_PATH)
    cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats


