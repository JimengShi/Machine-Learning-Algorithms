<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#XGBoost原理" data-toc-modified-id="XGBoost原理-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>XGBoost原理</a></span><ul class="toc-item"><li><span><a href="#提升方法（Boosting）" data-toc-modified-id="提升方法（Boosting）-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>提升方法（Boosting）</a></span><ul class="toc-item"><li><span><a href="#定义" data-toc-modified-id="定义-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>定义</a></span></li><li><span><a href="#加法模型" data-toc-modified-id="加法模型-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>加法模型</a></span></li><li><span><a href="#前向分步算法" data-toc-modified-id="前向分步算法-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>前向分步算法</a></span></li></ul></li><li><span><a href="#提升决策树-（BDT，Boosting-Decision-Tree）" data-toc-modified-id="提升决策树-（BDT，Boosting-Decision-Tree）-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>提升决策树 （BDT，Boosting Decision Tree）</a></span><ul class="toc-item"><li><span><a href="#定义" data-toc-modified-id="定义-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>定义</a></span></li><li><span><a href="#前向分步算法" data-toc-modified-id="前向分步算法-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>前向分步算法</a></span></li><li><span><a href="#回归问题的提升决策树" data-toc-modified-id="回归问题的提升决策树-1.2.3"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>回归问题的提升决策树</a></span></li><li><span><a href="#回归问题的提升决策树算法" data-toc-modified-id="回归问题的提升决策树算法-1.2.4"><span class="toc-item-num">1.2.4&nbsp;&nbsp;</span>回归问题的提升决策树算法</a></span></li></ul></li><li><span><a href="#梯度提升决策树-（GBDT，Gradient-Boosting-Decision-Tree）" data-toc-modified-id="梯度提升决策树-（GBDT，Gradient-Boosting-Decision-Tree）-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>梯度提升决策树 （GBDT，Gradient Boosting Decision Tree）</a></span><ul class="toc-item"><li><span><a href="#定义" data-toc-modified-id="定义-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>定义</a></span></li><li><span><a href="#梯度提升算法" data-toc-modified-id="梯度提升算法-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>梯度提升算法</a></span></li></ul></li><li><span><a href="#极限梯度提升（XGBoost，eXtreme-Gradient-Boosting）" data-toc-modified-id="极限梯度提升（XGBoost，eXtreme-Gradient-Boosting）-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>极限梯度提升（XGBoost，eXtreme Gradient Boosting）</a></span><ul class="toc-item"><li><span><a href="#定义" data-toc-modified-id="定义-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>定义</a></span></li><li><span><a href="#正则化目标函数" data-toc-modified-id="正则化目标函数-1.4.2"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>正则化目标函数</a></span></li><li><span><a href="#二阶泰勒展开" data-toc-modified-id="二阶泰勒展开-1.4.3"><span class="toc-item-num">1.4.3&nbsp;&nbsp;</span>二阶泰勒展开</a></span></li><li><span><a href="#分裂查找的精确贪婪算法" data-toc-modified-id="分裂查找的精确贪婪算法-1.4.4"><span class="toc-item-num">1.4.4&nbsp;&nbsp;</span>分裂查找的精确贪婪算法</a></span></li></ul></li></ul></li></ul></div>

# XGBoost
$$
\begin{align}
X\!G\!Boost&=eXtreme+GBDT\\
&=eXtreme+(Gradient+BDT) \\
&=eXtreme+Gradient+(Boosting+DecisionTree)
\end{align}
$$  

$$DecisionTree+Boosting \to BDT \to GBDT \to X\!G\!Boost$$



## XGBoost原理

### 提升方法（Boosting）

#### 定义
提升方法使用加法模型 + 前向分步算法。

#### 加法模型

<font color=red>**预测模型**</font>
$$f\left(x\right)=\sum_{m=1}^M\beta_m b\left(x;\gamma_m\right) \tag{1.1}$$

其中，$b\left(x;\gamma_m\right)$为基函数，比如线性回归，逻辑回归，决策树，神经网络等；$\gamma_m$为基函数的参数，$\beta_m$为基函数的系数。

<font color=red>**模型求解**</font>

在给定训练数据$\{\left(x_i,y_i\right)\}_{i=1}^N$及损失函数$L\left(y, f\left(x\right)\right)$的条件下，学习加法模型$f\left(x\right)$成为经验风险极小化问题：

$$\min_{\beta_m,\gamma_m}\sum_{i=1}^N L\left(y_i, \sum_{m=1}^M\beta_m b\left(x_i;\gamma_m\right)\right)\tag{1.2}$$

<font color=red>**模型问题**</font>

因为加法模型是由很多个模型累加而成的，比较难以优化，所以我们利用前向分步算法求解这一优化问题。其思路是：因为学习的是加法模型，可以从前向后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数式（1.2），则可以简化优化复杂度。具体地，每步只需优化如下损失函数：
$$\min_{\beta,\gamma}\sum_{i=1}^N L\left(y_i, \beta b\left(x_i;\gamma\right)\right)\tag{1.3}$$

#### 前向分步算法

输入：训练数据集$T=\{\left(x_1,y_1\right),\left(x_2,y_2\right),\dots,\left(x_N,y_N\right)\}$；损失函数$L\left(y,f\left(x\right)\right)$；基函数集合$\{b\left(x;\gamma\right)\}$；   
输出：加法模型$f\left(x\right)$  
（1）初始化$f_0\left(x\right)=0$  
（2）对$m=1,2,\dots,M$  
&emsp;&emsp;（a）极小化损失函数，以得到参数$\beta_m$，$\gamma_m$ 
$$\left(\beta_m,\gamma_m\right)=\mathop{\arg\min}_{\beta,\gamma} \sum_{i=1}^N L\left(y_i,  f_{m-1}\left(x_i\right)+\beta b\left(x_i;\gamma\right)\right) \tag{1.4}$$  

&emsp;&emsp;（b）更新
$$f_m\left(x\right)=f_{m-1}\left(x\right)+\beta_m b\left(x;\gamma_m\right) \tag{1.5}$$ 

（3）得到加法模型  
$$f\left(x\right)=f_M\left(x\right)=\sum_{m=1}^M\beta_m b\left(x;\gamma_m\right) \tag{1.6}$$

**总结：前向分步算法将<font color=red>同时求解</font>从$m=1$到$M$所有参数$\beta_m,\gamma_m$的优化问题简化为<font color=red>逐次求解</font>各个$\beta_m, \gamma_m$的优化问题。**

### 提升决策树 （BDT，Boosting Decision Tree）

#### 定义
以<font color=red>**决策树为基函数（基函数权重系数 = 1）**</font>的提升方法为提升决策树。提升决策树模型可以表示为决策树的加法模型：  

$$f_M=\sum_{m=1}^M T\left(x;\Theta_m\right) \tag{2.1}$$  
其中，$T\left(x;\Theta_m\right)$表示决策树；$\Theta_m$为决策树的参数；$M$为树的个数。

已知训练数据集$T=\{\left(x_1,y_1\right),\left(x_2,y_2\right),\dots\left(x_N,y_N\right)\}$，$x_i\in\mathcal{X}\subseteq\mathbb{R}^n$，$\mathcal{X}$为输入空间，$y_i\in\mathcal{Y}\subseteq\mathbb{R}$，$\mathcal{Y}$为输出空间。如果将输入空间$\mathcal{X}$划分为$J$个互不相交的区域$R_1,R_2,\dots,R_J$，并且在每个区域上确定输出的常量$c_j$，那么决策树可表示为
$$T\left(x;\Theta\right)=\sum_{j=1}^J c_j I\left(x\in R_j\right) \tag{2.4}$$
其中，参数$\Theta=\{\left(R_1,c_1\right),\left(R_2,c_2\right),\dots,\left(R_J,c_J\right)\}$表示决策树的区域划分和各区域上的常量值。$J$是决策树的复杂度即叶子结点个数。

<img src=./images/BDT.png width=60%>

#### 前向分步算法

提升决策树采用前向分步算法。首先确定初始提升决策树$f_0\left(x\right)=0$（只有一个节点的树），第$m$步的模型是

$$f_m\left(x\right)=f_{m-1}\left(x\right)+T\left(x;\Theta_m\right) \tag{2.2}$$

其中，$f_{m-1}\left(x\right)$为当前模型，通过经验风险极小化确定下一棵决策树的参数$\Theta_m$，
$$\hat{\Theta}_m=\mathop{\arg\min}_{\Theta_m}\sum_{i=1}^N L\left(y_i, f_{m-1}\left(x_i\right)+T\left(x_i;\Theta_m\right)\right) \tag{2.3}$$

提升决策树使用以下前向分步算法：
$$\begin{align}
f_0\left(x\right)&=0 \\
f_m\left(x\right)&=f_{m-1}\left(x\right)+T\left(x;\Theta_m\right),\quad m=1,2,\dots,M        \\
f_M\left(x\right)&=\sum_{m=1}^M T\left(x;\Theta_m\right)
\end{align}$$  
在前向分步算法的第$m$步，给定当前模型$f_{m-1}\left(x\right)$，需要求解  
$$\hat{\Theta}_m=\mathop{\arg\min}_{\Theta_m}\sum_{i=1}^N L\left(y_i,f_{m-1}\left(x_i\right)+T\left(x_i;\Theta_m\right)\right)$$
得到$\hat{\Theta}_m$，即第$m$棵树的参数。

#### 回归问题的提升决策树

当采用平方误差(MSE)损失函数时，
$$L\left(y,f\left(x\right)\right)=\left(y-f\left(x\right)\right)^2$$
其损失变为
$$\begin{align}
L\left(y,f_{m-1}\left(x\right)+T\left(x;\Theta_m\right)\right) 
&=\left[y-f_{m-1}\left(x\right)-T\left(x;\Theta_m\right)\right]^2 \\
&=\left[r-T\left(x;\Theta_m\right)\right]^2
\end{align}$$
其中，
$$r=y-f_{m-1}\left(x\right) \tag{2.5}$$

$r$是当前模型拟合数据的残差（residual），也是上一个模型没有学好的部分。对回归问题的提升决策树，只需要简单地拟合当前模型的残差。

#### 回归问题的提升决策树算法  
输入：训练数据集$T=\{\left(x_1,y_1\right),\left(x_2,y_2\right),\dots,\left(x_N,y_N\right)\}$；   
输出：提升决策树$f_M\left(x\right)$  
（1）初始化$f_0\left(x\right)=0$  
（2）对$m=1,2,\dots,M$ 

&emsp;&emsp;（a）按照式（2.5）计算残差
$$r_{mi}=y_i-f_{m-1}\left(x_i\right), \quad i=1,2,\dots,N$$  

&emsp;&emsp;（b）拟合残差$r_{mi}$学习一个回归树，得到$T\left(x;\Theta_m\right)$  

&emsp;&emsp;（c）更新$f_m\left(x\right)=f_{m-1}\left(x\right)+T\left(x;\Theta_m\right) $  

（3）得到回归提升决策树 
$$f_M\left(x\right)=\sum_{m=1}^M T\left(x;\Theta_m\right)   $$

### 梯度提升决策树 （GBDT，Gradient Boosting Decision Tree）

#### 定义

梯度提升算法使用损失函数的负梯度在当前模型的值
$$-\left[\frac{\partial L\left(y,f\left(x_i\right)\right)}{\partial f\left(x_i\right)}\right]_{f\left(x\right)=f_{m-1}\left(x\right)} \tag{3.1}$$

作为回归问题提升决策树算法中残差的近似值，拟合一个回归树。

#### 梯度提升算法  
输入：训练数据集$T=\{\left(x_1,y_1\right),\left(x_2,y_2\right),\dots,\left(x_N,y_N\right)\}$； 损失函数$L\left(y,f\left(x\right)\right)$  
输出：梯度提升决策树$\hat{f}\left(x\right)$  
（1）初始化
$$f_0\left(x\right)=\mathop{\arg\min}_c\sum_{i=1}^N L\left(y_i,c\right)$$
（2）对$m=1,2,\dots,M$  
&emsp;&emsp;（a）对$i=1,2,\dots,N$，计算
$$r_{mi}=-\left[\frac{\partial L\left(y,f\left(x_i\right)\right)}{\partial f\left(x_i\right)}\right]_{f\left(x\right)=f_{m-1}\left(x\right)}$$  
&emsp;&emsp;（b)对$r_{mi}$拟合一个回归树，得到第$m$棵树的叶结点区域$R_{mj},j=1,2,\dots,J$  
&emsp;&emsp;（c）对$j=1,2,\dots,J$，计算
$$c_{mj}=\mathop{\arg\min}_c\sum_{x_i\in R_{mj}} L\left(y_i, f_{m-1}\left(x_i\right)+c\right)$$  
&emsp;&emsp;（d）更新$f_m\left(x\right)=f_{m-1}\left(x\right)+\sum_{j=1}^J c_{mj} I\left(x\in R_{mj}\right)$  

（3）得到回归梯度提升决策树 
$$\hat{f}\left(x\right)=f_M\left(x\right)=\sum_{m=1}^M \sum_{j=1}^J c_{mj} I\left(x\in R_{mj}\right)$$

### 极限梯度提升（XGBoost，eXtreme Gradient Boosting）

#### 定义

训练数据集$\mathcal{D}=\{\left(\mathbf{x}_i,y_i\right)\}$, 其中$\mathbf{x}_i\in\mathbb{R}^m, y_i\in\mathbb{R}, {x}_i和{y}_i是列向量, \left|\mathcal{D}\right|=n$。

决策树模型
$$f\left(\mathbf{x}\right)=w_{q\left(\mathbf{x}\right)} \tag{4.1}$$

其中，$q:\mathbb{R}^m\to \{1,\dots,T\}$是有输入 $\mathbf{x}$ 向叶子结点编号的映射，$w\in\mathbb{R}^T$是叶子结点向量，$T$为决策树叶子节点数。

<img src=./images/XGBoost.png width=60%>

提升决策树模型预测输出
$$\hat{y}_i=\phi\left(\mathbf{x}_i\right)=\sum_{k=1}^K f_k\left(\mathbf{x}_i\right) \tag{4.2}$$
其中，$f_k\left(\mathbf{x}\right)$为第$k$棵决策树。

#### 正则化目标函数
$$\mathcal{L}\left(\phi\right)=\sum_i l\left(\hat{y}_i,y_i\right)+\sum_k \Omega\left(f_k\right) \tag{4.3}$$

其中，$\Omega\left(f\right)=\gamma T+\frac{1}{2}\lambda\|w\|^2=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^T w_j^2$，限制叶子节点个数 $T$，和模型输出 $w$。

**第$t$轮**目标函数
$$\mathcal{L}^{\left(t\right)}=\sum_{i=1}^n l\left(y_i, \hat{y}^{\left(t-1\right)}_i+f_t\left(\mathbf{x}_i\right)\right)+\Omega\left(f_t\right) \tag{4.4}$$

#### 二阶泰勒展开

第$t$轮目标函数$\mathcal{L}^{\left(t\right)}$在$\hat{y}^{\left(t-1\right)}$处的二阶泰勒展开
$$\begin{align}
\mathcal{L}^{\left(t\right)}&\simeq\sum_{i=1}^n\left[l\left(y_i,\hat{y}^{\left(t-1\right)}\right)+\partial_{\hat{y}^{\left(t-1\right)}}l\left(y_i,\hat{y}^{\left(t-1\right)}\right) f_t\left(\mathbf{x}_i\right)+\frac{1}{2}\partial^2_{\hat{y}^{\left(t-1\right)}}l\left(y_i,\hat{y}^{\left(t-1\right)}\right) f^2_t\left(\mathbf{x}_i\right)\right]+\Omega\left(f_t\right)  \tag{4.5}  \\
&=\sum_{i=1}^n\left[l\left(y_i,\hat{y}^{\left(t-1\right)}\right)+g_i f_t\left(\mathbf{x}_i\right)+\frac{1}{2}h_i f^2_t\left(\mathbf{x}_i\right)\right]+\Omega\left(f_t\right)
\end{align}$$

其中，一阶导数：$g_i=\partial_{\hat{y}^{\left(t-1\right)}}l\left(y_i,\hat{y}^{\left(t-1\right)}\right),  二阶导数： h_i=\partial^2_{\hat{y}^{\left(t-1\right)}}l\left(y_i,\hat{y}^{\left(t-1\right)}\right)$。

第$t$轮目标函数$\mathcal{L}^{\left(t\right)}$的二阶泰勒展开移除关于
$f_t\left(\mathbf{x}_i\right)$常数项
$$\begin{align}
\tilde{\mathcal{L}}^{\left(t\right)}&=\sum_{i=1}^n\left[g_i f_t\left(\mathbf{x}_i\right)+\frac{1}{2}h_i f^2_t\left(\mathbf{x}_i\right)\right]+\Omega\left(f_t\right)  \tag{4.6}\\
&=\sum_{i=1}^n\left[g_i f_t\left(\mathbf{x}_i\right)+\frac{1}{2}h_i f^2_t\left(\mathbf{x}_i\right)\right]+\gamma T+\frac{1}{2}\lambda\sum_{j=1}^T w_j^2
\end{align} \\$$  
定义叶结点$j$上的样本的下标集合$I_j=\{i|q\left(\mathbf{x}_i\right)=j\}$，则目标函数可表示为按叶结点累加的形式
$$\tilde{\mathcal{L}}^{\left(t\right)}=\sum_{j=1}^T\left[\left(\sum_{i\in I_j}g_i\right)w_j+\frac{1}{2}\left(\sum_{i\in I_j}h_i+\lambda\right)w_j^2\right]+\gamma T \tag{4.7}$$

<img src=./images/XGBoost1.png width=60%>

由于$$w_j^*=\mathop{\arg\min}_{w_j}\tilde{\mathcal{L}}^{\left(t\right)}$$
可令$$\frac{\partial\tilde{\mathcal{L}}^{\left(t\right)}}{\partial w_j}=0$$
得到每个叶结点$j$的最优分数为
$$w_j^*=-\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j} h_i+\lambda} \tag{4.8}$$

代入每个叶结点$j$的最优分数，得到最优化目标函数值
$$\tilde{\mathcal{L}}^{\left(t\right)}\left(q\right)=-\frac{1}{2}\sum_{j=1}^T 
\frac{\left(\sum_{i\in I_j} g_i\right)^2}{\sum_{i\in I_j} h_i+\lambda}+\gamma T \tag{4.9}$$

假设$I_L$和$I_R$分别为分裂后左右结点的实例集，令$I=I_L\cup I_R$，则分裂后损失减少量由下式得出

$$\mathcal{L}_{split}=\frac{1}{2}\left[\frac{\left(\sum_{i\in I_L} g_i\right)^2}{\sum_{i\in I_L}h_i+\lambda}+\frac{\left(\sum_{i\in I_R} g_i\right)^2}{\sum_{i\in I_R}h_i+\lambda}-\frac{\left(\sum_{i\in I} g_i\right)^2}{\sum_{i\in I}h_i+\lambda}\right]-\gamma \tag{4.10}$$
用以评估待分裂结点。

#### 分裂查找的精确贪婪算法  
输入：当前结点实例集$I$;特征维度$d$  
输出：根据最大分值分裂  
（1）$gain\leftarrow 0$  
（2）$G\leftarrow\sum_{i\in I}g_i$，$H\leftarrow\sum_{i\in I}h_i$  
（3）for $k=1$ to $d$ do  
（3.1）$G_L \leftarrow 0$，$H_L \leftarrow 0$  
（3.2）for $j$ in sorted($I$, by $\mathbf{x}_{jk}$) do  
（3.2.1）$G_L \leftarrow G_L+g_j$，$H_L \leftarrow H_L+h_j$  
（3.2.2）$G_R \leftarrow G-G_L$，$H_R=H-H_L$  
（3.2.3）$score \leftarrow \max\left(score,\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}\right)$  
（3.3）end  
（4）end
