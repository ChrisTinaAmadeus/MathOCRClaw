17. (本题满分 15 分)
如图，在三棱柱 $ABC-A_1B_1C_1$ 中，$CC_1\perp$ 平面 $ABC$，$AC=BC=CC_1=2$，$\angle ACB=90^\circ$，$D,E,M$ 分别是 $A_1B_1,CC_1,A_1B$ 的中点.
(1) 求证：$C_1D\parallel$ 平面 $A_1BE$；
(2) 求平面 $ABC$ 与平面 $A_1BE$ 所成角的余弦值.
<插图>

### 手写答案

(1) _未识别到手写答案。_
(2) 以 $C$ 为原点，$CA,CB,CC_1$ 分别为 $x,y,z$ 轴建立空间直角坐标系
$A(2,0,0)$ $B(0,2,0)$ $C(0,0,0)$ $A_1(2,0,2)$ $E(0,0,1)$
$\overrightarrow{AB}=(-2,2,0)$ $\overrightarrow{CB}=(0,2,0)$ $\overrightarrow{EA_1}=(2,0,1)$ $\overrightarrow{A_1B}=(-2,2,-2)$
设平面 $ABC$ 的一个法向量为 $\overrightarrow{n_1}$
$\begin{cases}\overrightarrow{AB}\cdot\overrightarrow{n_1}=0\\\overrightarrow{CB}\cdot\overrightarrow{n_1}=0\end{cases}$
令 $x=0$，则 $y=0$，$z=-4$，$\therefore \overrightarrow{n_1}=(0,0,-4)$
同理可得 $\overrightarrow{n_2}=(-1,1,2)$
$\therefore \cos\langle\overrightarrow{n_1},\overrightarrow{n_2}\rangle=\frac{|\overrightarrow{n_1}\cdot\overrightarrow{n_2}|}{|\overrightarrow{n_1}||\overrightarrow{n_2}|}=\frac{8}{\sqrt{16}\cdot\sqrt{6}}=\frac{8}{4\sqrt{6}}=\frac{2}{\sqrt{6}}=\frac{2\sqrt{6}}{6}=\frac{\sqrt{6}}{3}$

18. (本小题满分 17 分)
已知函数 $g(x)=2^x$，且函数 $f(x)$ 与 $g(x)$ 的图象关于直线 $y=x$ 对称.
(1) 求函数 $f(x)$ 的解析式；
(2) 若 $f(m^2+m)<1$ 成立，求实数 $m$ 的取值范围；
(3) 若 $0<a<b$ 且 $f(a)=-f(b)$，求 $f(a+b)$ 的取值范围.

### 手写答案

(1) $f(x)=\log_2 x$
(2) $f(m^2+m)<f(2)$
$\because f(x)$ 在 $(0,+\infty)$ 上 $\uparrow$
$\therefore m^2+m<2$ 且 $m^2+m>0$ $\Rightarrow$ $\begin{cases}-2<m<1\\m>0 \text{ 或 } m<-1\end{cases}$
$\therefore m$ 范围 $(-2,-1)\cup(0,1)$
(3) $\because f(a)=-f(b)$ 即 $\log_2 a=-\log_2 b$
$\therefore \log_2 a=\log_2\frac{1}{b}$ 即 $ab=1$
又 $\because 0<a<b$
$\therefore a+b>2\sqrt{ab}=2$
$\therefore f(a+b)=\log_2(a+b)>\log_2 2=1$
$\therefore f(a+b)$ 范围为 $(1,+\infty)$

19. (本小题满分 17 分)
已知函数 $f(x)=a\cdot 2^x-2^{-x}$ 是定义在 $\mathbf{R}$ 上的奇函数.
(1) 求实数 $a$ 的值，并证明：$f(x)$ 在 $\mathbf{R}$ 上单调递增；
(2) 求不等式 $f(3x^2-5x)+f(x-4)>0$ 的解集；
(3) 若 $g(x)=4^x+4^{-x}-2mf(x)$ 在区间 $[-1,+\infty)$ 上的最小值为 $-2$，求实数 $m$ 的值.

### 手写答案

(1) $\because f(x)=a\cdot 2^x-2^{-x}$ 是定义在 $\mathbf{R}$ 上的奇函数 $\therefore f(0)=0$
即 $a\cdot 2^0-2^0=0$ $\therefore a=1$
故 $f(x)=2^x-2^{-x}$
[定义法] 任取 $x_1<x_2\in\mathbf{R}$，
$f(x_1)-f(x_2)=2^{x_1}-\frac{1}{2^{x_1}}-(2^{x_2}-\frac{1}{2^{x_2}})$
$=(2^{x_1}-2^{x_2})+(\frac{1}{2^{x_2}}-\frac{1}{2^{x_1}})$
$=(2^{x_1}-2^{x_2})+\frac{2^{x_1}-2^{x_2}}{2^{x_1+x_2}}$
$=(2^{x_1}-2^{x_2})(1+\frac{1}{2^{x_1+x_2}})$
$\because x_1<x_2$ 故 $2^{x_1}<2^{x_2}$ 即 $2^{x_1}-2^{x_2}<0$，$1+\frac{1}{2^{x_1+x_2}}>0$
$\therefore f(x_1)-f(x_2)<0$ 即 $f(x_1)<f(x_2)$
$\therefore f(x)$ 在 $\mathbf{R}$ 上 $\uparrow$
(2) $\because f(x)$ 为奇函数 $\therefore f(x-4)=-f(4-x)$
$\therefore f(3x^2-5x)+f(x-4)>0$ 可化为 $f(3x^2-5x)>f(4-x)$
$f(x)$ 在 $\mathbf{R}$ 上 $\uparrow$
$\therefore 3x^2-5x>4-x$ 即 $3x^2-4x-4>0$
$x<-\frac{2}{3}$ 或 $x>2$
(3) $g(x)=4^x+4^{-x}-2m(2^x-2^{-x})$
即 $g(x)=(2^x-2^{-x})^2-2m(2^x-2^{-x})+2$
设 $t=2^x-2^{-x}$，由 $x\geq-1$ 则 $t\geq-\frac{3}{2}$
$\therefore y=t^2-2mt+2$ 在 $[-\frac{3}{2},+\infty)$ 上最小值为 $-2$
对称轴 $t=m$
当 $m<-\frac{3}{2}$ 时 $(-\frac{3}{2})^2-2m\cdot(-\frac{3}{2})+2=-2$，$m=-\frac{25}{12}$
当 $m\geq-\frac{3}{2}$ 时 $m^2-2m^2+2=-2$，$m=2$，$m=-2$（舍）
$\therefore m=-\frac{25}{12}$ 或 $m=2$