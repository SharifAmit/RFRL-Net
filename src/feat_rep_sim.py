def feat_rep_sim(f_1, f_2):
    def loss(y_true, y_pred,sample_weight):
        frs_loss = 0
        for i in range(len(f_1)):
            frs_loss += K.mean(K.abs(f_1[i] - f_2[i]))
        frs_loss *= sample_weight
        return frs_loss
