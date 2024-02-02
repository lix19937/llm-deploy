
# 算子融合     

## 有依赖的融合  
 减少了访存

从合并难度的角度，算子又被分为    
+ pointwise（elementwise）    
+ reduction  
+ 计算密集型，如matmul和convolution     

访存密集型     
elementwise + elementwise, elementwise + reduction, reduction + elementwise, reduction + reduction

A = relu(B + C)



## 无依赖的融合
 可能会减少launch kernel时间    


## sigmoid  
是 elementwise op   
单调函数   多标签分类  

## softmax   
是 reduction op     多类别分类    

一般方法是 3次kernel  `获得全局最大值 M= max(x_i)， S=sum_i(exp(x_i - M))， y_i = x_i /S`        
online方法 或 1-pass方法  

先把全部数据集中到一个线程块，然后在一个线程块内做规约，由此得到全局最大值呢？  
如果这种方法行得通，我们就可以马上利用相同的方法把数据全部集中到一个线程块，然后再一个线程块做sum规约得到全局求和，到此最后在接入更新向量的过程即可。   
整个过程理论上只需要一次kernel就可以完成，同时还可以减少内存的开销


## bn ln  
Welford  对数组一次遍历，即可同时得到mean和variance    

## Ref   
https://zhuanlan.zhihu.com/p/561627225   
https://zhuanlan.zhihu.com/p/672960528
