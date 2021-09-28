#!/usr/bin/python
# -*- coding: utf-8 -*-

def write2txt(outpath, V, trial, cla, algo, feat, threshold, train_equiv, kf_thre, \
                kf_acc_scores, kf_pre_scores, kf_rec_scores, kf_f1_scores, \
                nested_kf_thre, nested_kf_acc_scores, nested_kf_pre_scores, \
                nested_kf_rec_scores, nested_kf_f1_scores):
    # calculate score differences btw outer loop and inner loop
    acc_score_diff_kf = kf_acc_scores - nested_kf_acc_scores
    pre_score_diff_kf = kf_pre_scores - nested_kf_pre_scores
    rec_score_diff_kf = kf_rec_scores - nested_kf_rec_scores
    f1_score_diff_kf = kf_f1_scores - nested_kf_f1_scores

    # values to return to write latex tables
    values = []
    accs = []
    pres = []
    recs = []
    f1s = []
    
    if train_equiv:
        outdir = outpath + 'score_kf_nkf_v' + str(V) + '_equiv/'
    else:
        outdir = outpath + 'score_kf_nkf_v' + str(V) + '/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    def num2str(number):
        return '{} '.format(number).replace('.', ',')

    outf = os.path.join(outdir, cla + '_' + algo + '_' +
                        feat + '_thres_{}'.format(threshold) + '.txt')
    with open(outf, 'w') as outstream:
        for j, _feat in enumerate(feat.split('+')):
            for i in range(trial):
                outstream.write(num2str(kf_thre[i][j]))
            outstream.write('\n')
        outstream.write('\n')

        for i in range(trial):
            outstream.write(num2str(kf_acc_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(kf_acc_scores.mean()) + '\n\n')

        for i in range(trial):
            outstream.write(num2str(kf_pre_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(kf_pre_scores.mean()) + '\n\n')

        for i in range(trial):
            outstream.write(num2str(kf_rec_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(kf_rec_scores.mean()) + '\n\n')

        for i in range(trial):
            outstream.write(num2str(kf_f1_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(kf_f1_scores.mean()) + '\n\n')

        outstream.write('\n')
        #==============================================================
        nested_kf_thre = np.array(nested_kf_thre)
        for j, _feat in enumerate(feat.split('+')):
            for i in range(trial):
                outstream.write(num2str(nested_kf_thre[i, :, j].mean()))
                values.append(nested_kf_thre[i, :, j].mean())
            outstream.write('\n')
        outstream.write('\n')

        for i in range(trial):
            outstream.write(num2str(nested_kf_acc_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(nested_kf_acc_scores.mean()) + '\n\n')
        accs.append(nested_kf_acc_scores.mean())

        for i in range(trial):
            outstream.write(num2str(nested_kf_pre_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(nested_kf_pre_scores.mean()) + '\n\n')
        pres.append(nested_kf_pre_scores.mean())

        for i in range(trial):
            outstream.write(num2str(nested_kf_rec_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(nested_kf_rec_scores.mean()) + '\n\n')
        recs.append(nested_kf_rec_scores.mean())

        for i in range(trial):
            outstream.write(num2str(nested_kf_f1_scores[i]))
        outstream.write('\n')
        outstream.write(num2str(nested_kf_f1_scores.mean()) + '\n\n')
        f1s.append(nested_kf_f1_scores.mean())
        
        outstream.write('\n')
        #==============================================================
        for i in range(trial):
            outstream.write(num2str(acc_score_diff_kf[i]))
        outstream.write('\n')
        outstream.write(num2str(acc_score_diff_kf.mean()) + '\n')
        outstream.write(num2str(acc_score_diff_kf.std()) + '\n\n')

        for i in range(trial):
            outstream.write(num2str(pre_score_diff_kf[i]))
        outstream.write('\n')
        outstream.write(num2str(pre_score_diff_kf.mean()) + '\n')
        outstream.write(num2str(pre_score_diff_kf.std()) + '\n\n')

        for i in range(trial):
            outstream.write(num2str(rec_score_diff_kf[i]))
        outstream.write('\n')
        outstream.write(num2str(rec_score_diff_kf.mean()) + '\n')
        outstream.write(num2str(rec_score_diff_kf.std()) + '\n\n')

        for i in range(trial):
            outstream.write(num2str(f1_score_diff_kf[i]))
        outstream.write('\n')
        outstream.write(num2str(f1_score_diff_kf.mean()) + '\n')
        outstream.write(num2str(f1_score_diff_kf.std()) + '\n\n')
    return values, accs, pres, recs, f1s

