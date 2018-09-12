import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plotLLs(name,outfolder,xepochs,costs,log_px,log_pz,log_qz):
    plt.figure(figsize=[12,12])
    plt.plot(xepochs,costs, label="LL")
    plt.plot(xepochs,log_px, label="logp(x|z1)")
    for ii,p in enumerate(log_pz):
        plt.plot(xepochs,p, label="log p(z%i)"%ii)
    for ii,p in enumerate(log_qz):
        plt.plot(xepochs,p, label="log q(z%i)"%ii)
    plt.xlabel('Epochs'), plt.ylabel('log()'), plt.grid('on')
    plt.ylim([-150,0])
    plt.title(name), plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(outfolder+'/'+ name +'.png'),  plt.close()

def plotLLssemisub(name,outfolder,xepochs,costs,log_px,log_pz,log_qz):
    plt.figure(figsize=[12,12])
    for ii,c in enumerate(costs):
        plt.plot(xepochs,c, label="cost-%i"%ii)
    plt.plot(xepochs,log_px, label="logp(x|z1)")
    for ii,p in enumerate(log_pz):
        plt.plot(xepochs,p, label="log p(z%i)"%ii)
    for ii,p in enumerate(log_qz):
        plt.plot(xepochs,p, label="log q(z%i)"%ii)
    plt.xlabel('Epochs'), plt.ylabel('log()'), plt.grid('on')
    plt.title(name), plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(outfolder+'/'+ name +'.png'),  plt.close()

def plotKLs(name,outfolder,xepochs,KL,vmin=0,vmax=2):
    fig, ax = plt.subplots()
    data = np.concatenate(KL,axis=0).T
    heatmap = ax.pcolor(data, cmap=plt.cm.Greys,  vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_xticklabels(xepochs, minor=False)
    plt.xlabel('Epochs'), plt.ylabel('#Latent Unit'), plt.title('KL(q|p)'), plt.colorbar(heatmap)
    plt.savefig(outfolder+'/' + name +'.png'),  plt.close()

def boxplot(res_out,name,data):
    #name = 'boxplot:mu_q_iw%i'%j
    #data = var_p[0]
    bs,eq,iw,nf = data.shape
    data_pl = [data.mean(axis=(1,2))[:,n] for n in range(nf)]
    fig, ax = plt.subplots()
    plt.boxplot(data_pl)
    plt.ylim([-12,12])
    plt.xlabel('Unit (min/max)')
    plt.grid('on')
    ticks = ["%0.0e/%0.0e"%(d.min(),d.max()) for d in data_pl]
    plt.xticks(range(1,len(data_pl)+1),ticks,fontsize = 5)
    plt.savefig(res_out+'/' + name +'.png')

def read_from_log(logfile):
    train_loss_x = []
    train_loss_y = []
    test_loss_x = []
    test_loss_y = []

    train_tag = '====> Epoch:'
    test_tag = '====> Test set loss:'
    with open(logfile,'r') as f:
        for line in f:
            if line.startswith(train_tag):
                epoch = int(line.split(train_tag)[1].strip().split(' ')[0])
                loss = float(line.split('Average loss:')[1].strip().split(' ')[0])
                
                if epoch < 200:
                    continue
                train_loss_x.append(epoch)
                train_loss_y.append(loss)
            elif line.startswith(test_tag):
                loss = float(line.split(test_tag)[1].strip().split(' ')[0])

                if epoch < 200:
                    continue
                test_loss_x.append(epoch)
                test_loss_y.append(loss)

    return np.asarray(train_loss_x), np.asarray(train_loss_y), np.asarray(test_loss_x), np.asarray(test_loss_y)

def plotTSNE(figname,z,target, name, legend=False):
    fig, ax = plt.subplots()
    #pca = PCA(n_components=2)
    pca = TSNE(n_components=2)
    X_r = pca.fit_transform(z)

    plt.figure()
    if target is not None:
        num_class = len(np.unique(target))
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_class))
        for i,c in zip(range(num_class),colors):
            plt.scatter(X_r[target == i, 0], X_r[target == i, 1], c=c, label=str(i),s=8,alpha=1.0)
    else:
        plt.scatter(X_r[:, 0], X_r[:, 1],s=5)

    ax.grid('off')
    ax.set_aspect('equal')                
    ax.set_xlabel(name)
    ax.xaxis.set_label_position('top') 

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        labelleft='off',
        labelbottom='off'
    )

    #plt.ylim([-4,4])
    if legend:
        lgnd = plt.legend([str(i) for i in range(num_class)], bbox_to_anchor=(1.00, 0.5), frameon=False, loc='center left', prop={'size': 18})
        for handle in lgnd.legendHandles:
            handle.set_sizes([20])

    plt.show()
    plt.savefig(figname, bbox_inches='tight'),  plt.close()


def plotTSNE_all(figname, zs, target, x_names, y_names, nrow):
    tsne = TSNE(n_components=2)
    #tsne = PCA(n_components=2) 

    size = len(zs)
    ncol = size / nrow
    
    if target is not None:
        num_class = len(np.unique(target))
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_class))

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol)

    index = 0

    if nrow==1:
        ax = [ax]

    for ii, row in enumerate(ax):
        for jj, pt in enumerate(row):
            z = zs[index]
            X_r = tsne.fit_transform(z)

            if target is not None:
                for i,c in zip(range(num_class),colors):
                    pt.scatter(X_r[target == i, 0], X_r[target == i, 1], c=c, label=str(i),s=2,alpha=.5)
            else:
                pt.scatter(X_r[:, 0], X_r[:, 1],s=2)


            pt.set_xlabel(x_names[index])    
            pt.xaxis.set_label_position('top') 

            #if jj==0:
            #pt.set_ylabel(y_names[ii], rotation=0, ha='right')
            
            pt.set_aspect('equal')                
            pt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                left='off',
                labelleft='off',
                labelbottom='off')

            index += 1

    # need to fix legend
    #plt.legend([str(i) for i in range(num_class)], loc='best', bbox_to_anchor=(1.05, 0), frameon=False, fancybox=True)
    plt.show()
    plt.savefig(figname, bbox_inches='tight'),  plt.close()
    
def plot_learning_curve(figname, loss_list, names):
    size = len(loss_list)
    plt.figure()

    #colors = matplotlib.cm.rainbow(np.linspace(0, 1, size))
    colors = plt.cm.get_cmap('Accent').colors
    #colors = matplotlib.cm('Accent')

    for i, loss in enumerate(loss_list):
        train_x, train_y, test_x, test_y = loss
        plt.plot(train_x, -train_y, c=colors[i], label=names[i])
        plt.plot(test_x, -test_y, '--', c=colors[i])

    if names is not None:
        plt.legend(names)

    plt.show()
    plt.xlim([200,2000])
    plt.xlabel('Epoch')
    plt.ylabel('LL', rotation=0, ha='right')
    plt.savefig(figname), plt.close()
        
