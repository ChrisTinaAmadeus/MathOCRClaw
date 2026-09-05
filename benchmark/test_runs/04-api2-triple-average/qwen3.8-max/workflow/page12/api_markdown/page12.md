17. 如图，在三棱柱 $ABC-A_1B_1C_1$ 中，$B_1B \perp$ 平面 $A_1B_1C_1$，$AC=CB=CC_1=2$，$\angle ACB=90^\circ$，
$D$，$E$ 分别是 $A_1B_1$，$CC_1$ 的中点。
(1) 求证：$C_1D \parallel$ 平面 $A_1BE$；
(2) 求平面 $ABC$ 与平面 $A_1BE$ 所成角的正弦值。
<插图>

### 手写答案

证：(1) ∵ 在三棱柱 $ABC-A_1B_1C_1$ 中，
$B_1B \perp$ 平面 $A_1B_1C_1$
∴ 平面 $ABC \parallel$ 平面 $A_1B_1C_1$，
$B_1B \parallel C_1C$
∴ $C_1C \perp$ 平面 $ABC$
如图建立空间直角坐标系
∴ $C_1(0,0,2)$，$D(1,1,2)$，$A_1(2,0,2)$
$B(0,2,0)$，$E(0,0,1)$
∴ $\overrightarrow{C_1D}=(1,1,0)$，$\overrightarrow{A_1B}=(-2,2,-2)$
$\overrightarrow{A_1E}=(-2,0,-1)$
设平面 $A_1BE$ 一个法向量为 $\overrightarrow{n_1}=(x,y,z)$
则 $\begin{cases}\overrightarrow{n_1}\cdot\overrightarrow{A_1B}=0\\ \overrightarrow{n_1}\cdot\overrightarrow{A_1E}=0\end{cases}$ $\Rightarrow$ $\begin{cases}-2x+2y-2z=0\\ -2x-z=0\end{cases}$
令 $x=-1$，则 $y=1$，$z=2$
∴ $\overrightarrow{n_1}=(-1,1,2)$
∴ $\overrightarrow{C_1D}\cdot\overrightarrow{n_1}=1\times(-1)+1\times1+0\times2=0$
∴ $\overrightarrow{C_1D} \perp \overrightarrow{n_1}$
∴ $C_1D \parallel$ 平面 $A_1BE$
(2) 由 (1) 得平面 $A_1BE$ 的法向量 $\overrightarrow{n_1}=(-1,1,2)$
$A(2,0,0)$，$B(0,2,0)$，$C(0,0,0)$
∴ $\overrightarrow{CA}=(2,0,0)$，$\overrightarrow{CB}=(0,2,0)$
设平面 $ABC$ 一个法向量为 $\overrightarrow{m}=(a,b,c)$
则 $\begin{cases}\overrightarrow{m}\cdot\overrightarrow{CA}=0\\ \overrightarrow{m}\cdot\overrightarrow{CB}=0\end{cases}$ $\Rightarrow$ $\begin{cases}2a=0\\ 2b=0\end{cases}$
令 $a=0$，则 $b=0$，$c=1$
∴ $\overrightarrow{m}=(0,0,1)$
∴ $\overrightarrow{n_1}\cdot\overrightarrow{m}=-1\times0+1\times0+2\times1=2$
$|\overrightarrow{n_1}|=\sqrt{(-1)^2+1^2+2^2}=\sqrt{6}$，$|\overrightarrow{m}|=\sqrt{0^2+0^2+1^2}=1$
∴ $\cos\langle\overrightarrow{n_1},\overrightarrow{m}\rangle=\frac{\overrightarrow{n_1}\cdot\overrightarrow{m}}{|\overrightarrow{n_1}|\cdot|\overrightarrow{m}|}=\frac{2}{\sqrt{6}\times1}=\frac{\sqrt{6}}{3}$
设平面 $ABC$ 与平面 $A_1BE$ 所成角为 $\theta$
∴ $\cos\theta=|\cos\langle\overrightarrow{n_1},\overrightarrow{m}\rangle|=\frac{\sqrt{6}}{3}$
∴ $\sin\theta=\sqrt{1-\cos^2\theta}=\frac{\sqrt{3}}{3}$
答：正弦值为 $\frac{\sqrt{3}}{3}$