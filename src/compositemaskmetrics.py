import matplotlib.pyplot as plt
from CompositeMask import CompositeMask

#use OpenCV create a boolean mask by identifying all blue areas of the image
#true blue is [0, 0, 255]; we want to catch a range just in case
def make_boolean_mask(segmented_img):
    low_bg = np.array([0, 0, 250])
    high_bg = np.array([5, 5, 255])
    return cv2.inRange(segmented_img, low_bg, high_bg)

def cleanup_comp_mask(comp_mask):
    temp = np.array([255 if p > 127 else 0 for p in comp_mask.flatten()])
    return temp.reshape(300, 200)


def build_comp_image_and_confusion_dict(image_mask, manual_mask):
    '''
    compares predictions (composite mask) to truth (segmented mask)
    returns a new image with 4 levels: 255 for true background predictions
                                       200 for figure predicted to be background
                                       100 for background predicted to be figure
                                       0 for true figure predictions

    a dict represention of a confusion matrix
    '''

    confusion_dict = {'true_bg_pred_bg': 0, 'true_bg_pred_fig': 0, 'true_fig_pred_bg': 0, 'true_fig_pred_fig': 0}
    new_image = []
    for row_manual, row_cm in zip(manual_mask, image_mask):
        for item_manual, item_cm in zip(row_manual, row_cm):
            if item_manual == item_cm :
                if item_manual == 255:
                    new_image.append(255)
                    confusion_dict['true_bg_pred_bg'] += 1
                else:
                    new_image.append(0)
                    confusion_dict['true_fig_pred_fig'] += 1
            else:
                if item_manual > 200:
                    new_image.append(100)
                    confusion_dict['true_bg_pred_fig'] += 1
                else:
                    new_image.append(200)
                    confusion_dict['true_fig_pred_bg'] += 1

    return np.array(new_image).reshape(300, 200), confusion_dict

def get_comparison_metrics(confusion_dict):
    accuracy = (confusion_dict['true_bg_pred_bg'] + confusion_dict['true_fig_pred_fig']) / 60000.0
    precision_of_fig = (confusion_dict['true_fig_pred_fig'] + 1)*1. / \
            (confusion_dict['true_bg_pred_fig'] + confusion_dict['true_fig_pred_fig'] + 1)
    recall_of_fig = (confusion_dict['true_fig_pred_fig'] + 1)*1. / \
        (confusion_dict['true_fig_pred_bg'] + confusion_dict['true_fig_pred_fig'] + 1)
    harmonic_mean = 2.*(precision_of_fig * recall_of_fig) / (precision_of_fig + recall_of_fig)
    string_out = 'accuracy: ' + str(accuracy) + '\n'
    string_out += 'precision: ' + str(precision_of_fig) + '\n'
    string_out += 'recall: ' + str(recall_of_fig) + '\n'
    string_out += 'harmonic mean: ' + str(harmonic_mean)
    return accuracy, precision_of_fig, recall_of_fig, harmonic_mean, string_out

def from_paths_to_conf_dict(path_to_orig, path_to_manual_seg, threshold, num_bg_pts):

    fig = plt.figure(figsize=(8, 3))

    #create composite mask object
    cm = CompositeMask(path_to_orig, threshold, num_bg_pts)
    comp_mask = cleanup_comp_mask(cm.full_composite_mask)
    fig.add_subplot(131)
    plt.imshow(comp_mask)
    #plt.show()


    man_seg_img = CompositeMask(path_to_manual_seg).image
    manual_seg_mask = make_boolean_mask(man_seg_img)
    fig.add_subplot(132)
    plt.imshow(manual_seg_mask)
    #plt.show()

    truth_img, conf_dict = build_comp_image_and_confusion_dict(comp_mask, manual_seg_mask)
    acc, prec, rec, harm, string_out = get_comparison_metrics(conf_dict)
    fig.add_subplot(133)
    plt.imshow(truth_img)
    plt.show()

    print(string_out)

    return acc, prec, rec, harm
