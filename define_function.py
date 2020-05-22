#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:09:58 2020

@author: jingsixu
"""
import numpy as np
import pandas as pd

#‘lbfgs’, ‘liblinear’
def logitR(penalty,C,solver,X_train, y_train, X_cols):
    np.random.seed(123)
    # fit model
    logmodel = LogisticRegression(penalty=penalty,C=C,solver=solver,max_iter=1000,tol=1e-5)
    logmodel.fit(X_train, y_train)

    #coefficient
    coef_sk = pd.DataFrame(logmodel.coef_).T
    coef_sk['feature'] = X_cols
    coef_sk.set_index('feature',inplace=True)

    #prediction
    #predictions = logmodel.predict(X_test)
    return logmodel,coef_sk

def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return coefs, p 

def feature(model, X_train):
    series = pd.Series(model.feature_importances_, index=X_train.columns).nlargest(len(X_train.columns))
    series.plot(kind='barh')
    plt.title("Feature Importance", fontsize=15)
    plt.show()

def probhist(prob, y_test):
    '''
    generate final score for KS calculation
    '''
    (n, bins, p) = plt.hist(prob, bins=20)
    plt.show()

    #create KS measure table
    lower_score = bins[0:20]
    upper_score = bins[1:21]

    #create final_score table for model performance measure
    final_score = {'final_score': prob, 'd90_flag': y_test}
    final_score = pd.DataFrame(final_score)
    return lower_score, upper_score, final_score

def KS(lower_score, upper_score, final_score):
    
    #seperate dataset into default and nondefault dataset
    default = final_score[final_score['d90_flag']== 1]
    nondefault = final_score[final_score['d90_flag']== 0]


    KS_measure = pd.DataFrame({'lower_score': lower_score, 'upper_score': upper_score })

    #create number of default and non-default columns
    def countdef(a):
        count = 0
        for i in default['final_score']:
            if (i < a[1]) & (i > a[0]):
                count +=1
        return count

    def countnondef(a):
        count = 0
        for i in nondefault['final_score']:
            if (i < a[1]) & (i > a[0]):
                count +=1
        return count

    KS_measure['num_default'] = KS_measure[['lower_score','upper_score']].apply(countdef,axis=1)
    KS_measure['num_nondefault'] = KS_measure[['lower_score','upper_score']].apply(countnondef,axis=1)

    KS_measure['default_rate'] = KS_measure['num_default'].apply(lambda x: x/default.shape[0])
    KS_measure['nondefault_rate'] = KS_measure['num_nondefault'].apply(lambda x: x/nondefault.shape[0])

    KS_measure['cum_default'] = KS_measure['default_rate'].cumsum()
    KS_measure['cum_nondefault'] = KS_measure['nondefault_rate'].cumsum()

    KS_measure['K_S'] = KS_measure['cum_default'] - KS_measure['cum_nondefault']

    KS_max = round(KS_measure['K_S'].max()*100,1)

    KSmax_score = float(KS_measure[KS_measure['K_S']==KS_measure['K_S'].max()]['lower_score'])

    #visualize the KS measure
    plt.plot(KS_measure['lower_score'], KS_measure['cum_default'], color='red', label='default')
    plt.plot(KS_measure['lower_score'], KS_measure['cum_nondefault'], color='blue', label='non-default')
    plt.legend()  

    plt.xlabel('score')
    plt.ylabel('Percentage of Loans')
    plt.vlines(KSmax_score,0,1,linestyles='dashdot')

    plt.annotate("K-S max:%5.1f"%(KS_max)+"%", 
                 xy=(KSmax_score, 0.6), xytext=(KSmax_score+20, 0.6),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
    plt.title('K_S Measure, K_S max=%5.1f'%KS_max)
    plt.show()
    
    #print(KS_measure)
    
    return KS_max


##auc##########################################################################
def auc(model, X_test, y_test):
    y_score = model.predict_proba(X_test)
    y_true = y_test
    auc = roc_auc_score(y_true, y_score[:,1])
    return auc 

def auc_xgb(model, X_test, y_test):
    y_score = model.predict(X_test)
    y_true = y_test
    auc = roc_auc_score(y_true, y_score)
    return auc



##roc##########################################################################
def ROC(model, X_test, y_test, auc_model):
    gbrt_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, color = 'r', label='Model (area = %0.4f)' % auc_model)
    plt.plot([0, 1], [0, 1],'b--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC', fontsize=15)
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    plt.show()
    for i in range(len(fpr)):
        if abs(fpr[i] - 0.2) < 1e-2:
            print ('When fpr = 0.2, tpr = ', tpr[i])
            break
    opt_idx = np.argmin(np.sqrt(np.square(1-tpr) + np.square(fpr)))
    print('Best Threshold=%f' % (thresholds[opt_idx ]))
    return thresholds[opt_idx ], tpr, fpr

 def ROC_xgb(model, X_test, y_test, auc_model):
        gbrt_roc_auc = roc_auc_score(y_test, model.predict(X_test))
        fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
        plt.figure()
        plt.plot(fpr, tpr, color = 'r', label='Model (area = %0.4f)' % auc_model)
        plt.plot([0, 1], [0, 1],'b--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC', fontsize=15)
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
        plt.show()
        for i in range(len(fpr)):
            if abs(fpr[i] - 0.2) < 1e-2:
                print ('When fpr = 0.2, tpr = ', tpr[i])
            break

        opt_idx = np.argmin(np.sqrt(np.square(1-tpr) + np.square(fpr)))
        print('Best Threshold=%f' % (thresholds[opt_idx ]))
        return thresholds[opt_idx ], tpr, fpr


##classification report##########################################################################
def cp(model, X_test, y_test, threshold): 
    y_prob = model.predict_proba(X_test)[:,1]
    predict_mine = np.where(y_prob > threshold, 1, 0)
    print(classification_report(y_test, predict_mine))
    print('recall=%0.5f' % recall_score(y_test, predict_mine))
    print('precision=%0.5f' % precision_score(y_test, predict_mine))
    print('f1_score=%0.5f' % f1_score(y_test, predict_mine))
    cm = confusion_matrix(y_test,predict_mine) 
    return y_prob, cm

def cp_xgb(model, X_test, y_test, threshold): 
    y_prob = model.predict(X_test)
    predict_mine = np.where(y_prob > threshold, 1, 0)
    print(classification_report(y_test, predict_mine))  
    print('recall=%0.5f' % recall_score(y_test, predict_mine))
    print('precision=%0.5f' % precision_score(y_test, predict_mine))
    print('f1_score=%0.5f' % f1_score(y_test, predict_mine))
    cm = confusion_matrix(y_test,predict_mine) 
    return y_prob, cm


##confusion matrix##########################################################################
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    thresh = cm.max() / 2.
    for i, j in itt.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",  fontsize=12
                 )
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)


##put all measurement together##########################################################################
def run_all(model, X_train, y_train, X_test, y_test):
    
    ### frature importance
    feature_model = feature(model, X_train)
    
    ### K-S Test
    y_pred_model = model.predict(X_test)
    log_prob_model = -(model.predict_log_proba(X_test)[:, 1] )*10
    lower_score, upper_score, final_score = probhist(log_prob_model, y_test)
    ks_model = KS(lower_score, upper_score, final_score)

    ### AUC
    auc_model = auc(model, X_test, y_test)
    print(auc_model)

    ### ROC
    best_threshold, tpr, fpr = ROC(model, X_test, y_test, auc_model)

    ### Threshold from Logit Percentage
    y_prob = model.predict_proba(X_test)[:,1]
    y_prob_sort = np.sort(y_prob)
    plt.plot(y_prob_sort)
    plt.show()
    
    index = 15751
    threshold_index = y_prob_sort[-(index+1)]
    y_prob, cm = cp(model, X_test, y_test, threshold_index)

    plot_confusion_matrix(cm, classes=[0,1], title='Confusion matrix')
    
    return y_prob


##swapset##########################################################################
def swap_set(y_pred_model):
    
    #Logistic Prediction
    opt_prob = 0.1108941788607104
    predict_mine = np.where(y_prob_logit > opt_prob, 1, 0)
    y_pred_logit = predict_mine
    
    #Combine predict_y of two models
    swap = {'y_pred_logit':y_pred_logit, 'y_pred_model':y_pred_model }
    swap = pd.DataFrame(swap)

    #calculate swap-in and swap-out population
    swap_in = swap[(swap['y_pred_logit']==1) & (swap['y_pred_model']==0)]
    swap_out = swap[(swap['y_pred_logit']==0) & (swap['y_pred_model']==1)]
    approved = swap[(swap['y_pred_logit']==0) & (swap['y_pred_model']==0)]
    declined = swap[(swap['y_pred_logit']==1) & (swap['y_pred_model']==1)]
    Total_approved = len(approved) + len(swap_out)
    Total_declined = len(declined) + len(swap_in)
    Total_approved_gbrt = len(approved) + len(swap_in)
    Total_declined_gbrt = len(declined) + len(swap_out)
    Total = Total_approved + Total_declined

    #Swap Sets Matrix
    swap_matrix_model = np.matrix([[len(approved), len(swap_out), Total_approved], 
                        [len(swap_in), len(declined), Total_declined],
                        [Total_approved_gbrt, Total_declined_gbrt, Total]])
    print('Swap Sets Matrix')
    print(swap_matrix_model)
    
    #swap_graph = plot_confusion_matrix(swap_matrix_model, classes=['Approved','Declined','Total'], title='Swap Sets Matrix')
   
    
    return swap_matrix_model#, swap_graph



##badrate##########################################################################
def badrate(gbrt_pred_y, sawp_matrix, y_test):
    
    opt_prob = 0.1108941788607104
    predict_mine = np.where(y_prob_logit > opt_prob, 1, 0)
    y_pred_logit = predict_mine
   
    #combine actual default and predicted default from two models
    total_y = {'y_test': y_test, 
               'y_pred_logit':y_pred_logit, 
               'gbrt_pred_y':gbrt_pred_y}
    total_y = pd.DataFrame(total_y)

    #calculate bad rates
    logit_approved_totalbad = len(total_y[(total_y['y_test']==1) & (total_y['y_pred_logit']==0)])
    logit_declined_totalbad = len(total_y[(total_y['y_test']==1) & (total_y['y_pred_logit']==1)])
    new_approved_totalbad = len(total_y[(total_y['y_test']==1) & (total_y['gbrt_pred_y']==0)])
    new_declined_totalbad = len(total_y[(total_y['y_test']==1) & (total_y['gbrt_pred_y']==1)])
    both_approved_bad = len(total_y[(total_y['y_test']==1) & (total_y['y_pred_logit']==0)& (total_y['gbrt_pred_y']==0)])
    both_declined_bad = len(total_y[(total_y['y_test']==1) & (total_y['y_pred_logit']==1)& (total_y['gbrt_pred_y']==1)])
    swapin_bad = len(total_y[(total_y['y_test']==1) & (total_y['y_pred_logit']==1)& (total_y['gbrt_pred_y']==0)])
    swapout_bad = len(total_y[(total_y['y_test']==1) & (total_y['y_pred_logit']==0)& (total_y['gbrt_pred_y']==1)])
    totalbad = len(total_y[(total_y['y_test']==1)])
    badnumber_matrix = np.array([[both_approved_bad, swapout_bad, logit_approved_totalbad], 
                            [swapin_bad, both_declined_bad, logit_declined_totalbad],
                            [new_approved_totalbad, new_declined_totalbad,totalbad ]])

    badrate_matrix = np.around(badnumber_matrix/sawp_matrix, decimals=5)*100
    print('Bad Rates Matrix (in %)')
    print(badrate_matrix)
    
    #badrate_graph = plot_confusion_matrix(badrate_matrix, classes=['Approved','Declined','Total'], title='Bad Rate Matrix')
    return badrate_matrix#, badrate_graph

##put all measurement together##########################################################################
    
















