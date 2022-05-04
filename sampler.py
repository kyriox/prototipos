import numpy as np 
from scipy.spatial.distance import *
from scipy.stats import  pearsonr
import faiss,sys
from sklearn.cluster import KMeans as skmeans
from sklearn.metrics import pairwise_distances
#from classifiers import kNN

def  pearsond(x,y):
     return 1-np.abs(pearsonr(x,y)[0])

class Initializer:
     def FFT(self):
         current=np.random.randint(0, self.data.shape[0])
         dist=pairwise_distances(self.data,[self.data[current]])
         mm=np.argmax(dist)
         order,distances=[current],[dist[mm]]
         self.centers_={current:self.data[current]}
         while len(self.centers_)<self.n_clusters:
             current=mm
             order.append(current)
             dist=np.minimum(dist, pairwise_distances(self.data,[self.data[current]])) #[d if d<nd else nd for d,nd in zip(dist,ndist)]
             mm=np.argmax(dist)
             distances.append(dist[mm])
             self.centers_[current]=self.data[current]    
         self.selection_order_=np.array(order)
         self.sample_labels_=np.array([])
         if len(self.true_labels):
             self.sample_labels_=self.true_labels[order]
         self.distances_=np.array(distances)
         return self
         

     def KMPP(self):
          idx=np.random.randint(self.data.shape[0],size=1)[0]
          self.centers_={}
          self.distances_=[]
          self.centers_[idx]=self.data[idx]
          D=pairwise_distances(self.data,[self.data[idx]])
          sample_labels=[]
          if len(self.true_labels):
                    sample_labels.append(self.true_labels[idx])
          for i in range(self.n_clusters-1):
               cp=np.cumsum(D/np.sum(D))
               r=np.random.rand()
               s=np.argwhere(cp>=r)[0][0]
               self.centers_[s]=self.data[s]
               dist=pairwise_distances(self.data, [self.data[s]])
               D=np.minimum(D,dist)
               if len(self.true_labels):
                    sample_labels.append(self.true_labels[s])
          self.sample_labels_=np.array(sample_labels)
          return self

     def Random(self):
         idx=np.random.permutation(self.data.shape[0])[:self.n_clusters]
         centers=self.data[idx,:]
         self.centers_=dict(zip(idx,centers))
         self.sample_labels_=np.array([])
         if len(self.true_labels):
             self.sample_labels_=self.true_labels[idx]
         return self
     
     def _make_index(self,d):
          self.ndata=np.ascontiguousarray(self.data.astype('float32'))
          self.index= faiss.IndexFlatL2(d) # indice que utiliza L2
          self.index=faiss.IndexIDMap(self.index)
          ids=np.array([i for i in range(len(self.ndata))])
          self.index.add_with_ids(self.ndata,ids)

     def DNet(self):
          n,d=self.data.shape
          self._make_index(d)
          self.centers_={}
          nn=int(len(self.data)/self.n_clusters)
          no_selected=[i for i in range(n)]
          sample_labels=[]
          while len(self.centers_)<self.n_clusters and len(no_selected):
               i=np.random.choice(no_selected,1)[0]
               self.centers_[i]=self.data[i]
               dd,ids=self.index.search(self.ndata[i:i+1],nn)
               self.index.remove_ids(ids[0])
               if len(self.true_labels):
                    sample_labels.append(self.true_labels[i])
               no_selected=np.setdiff1d(no_selected,ids)
          self.sample_labels_=np.array(sample_labels)
          return self     
         
     def fit(self,data,true_labels=[]):
         self.data=data
         self.true_labels=true_labels
         self.algorithm()
         return self
           
     def __init__(self,n_samples=3,algorithm='FFT'):
         self.n_clusters=n_samples
         self.algorithm=getattr(self, algorithm)

                                
    

         
