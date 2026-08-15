17. 如图，在三棱柱 $ABC-A_1B_1C_1$ 中，$B_1B \perp$ 平面 $A_1B_1C_1$，$AC = CB = CC_1 = 2$，$\angle ACB = 90°$，$D, E$ 分别是 $A_1B_1, CC_1$ 的中点。
<插图>
(1) 求证：$C_1D \parallel$ 平面 $A_1BE$；
(2) 求平面 $ABC$ 与平面 $A_1BE$ 所成角的正弦值。

### 手写答案

(1) $\because$ 在三棱柱 $ABC-A_1B_1C_1$ 中，$BB_1 \perp$ 平面 $A_1B_1C_1$，$\therefore$ 平面 $ABC \parallel$ 平面 $A_1B_1C_1$，$BB_1 \parallel CC_1$，$\therefore C_1C \perp$ 平面 $ABC$。如图，建立空间直角坐标系。
$\therefore C_1(0,0,2)$，$D(1,1,2)$，$A_1(2,0,2)$，$B(0,2,0)$，$E(0,0,1)$
$\therefore \overrightarrow{C_1D} = (1,1,0)$，$\overrightarrow{A_1B} = (-2,2,-2)$，$\overrightarrow{AE} = (-2,0,-1)$
设平面 $A_1BE$ 一个法向量为 $\vec{n} = (x,y,z)$，则 $\vec{n} \cdot \overrightarrow{A_1B} = 0$，$\vec{n} \cdot \overrightarrow{AE} = 0 \Rightarrow -2x+2y-2z=0$，$-2x-z=0$
令 $x=-1$，则 $y=1$，$z=2$
$\therefore \vec{n} = (-1,1,2)$
$\therefore \overrightarrow{C_1D} \cdot \vec{n} = 1 \times (-1)+1 \times 1+0 \times 2 = 0$
$\therefore \overrightarrow{C_1D} \perp \vec{n}$，$\because C_1D \parallel$ 平面 $A_1BE$
$\therefore C_1D \parallel$ 平面 $A_1BE$

(2) 由 (1) 得平面 $A_1BE$ 的法向量 $\vec{n} = (-1,1,2)$；$A(2,0,0)$，$B(0,2,0)$，$C(0,0,0)$，$\therefore\overrightarrow{CA} = (2,0,0)$，$\overrightarrow{CB} = (0,2,0)$
设平面 $ABC$ 一个法向量为 $\vec{m} = (a,b,c)$，则 $\vec{m} \cdot \overrightarrow{CA} = 0$，$\vec{m} \cdot \overrightarrow{CB} = 0 \Rightarrow 2a=0$，$2b=0$；令 $a=0$，则 $b=0$，$c=1$
$\therefore \vec{m} = (0,0,1)$
$\because \vec{n} \cdot \vec{m} = -1 \times 0+1 \times 0+2 \times 1 = 2$，$|\vec{n}| = \sqrt{(-1)^2+1^2+2^2} = \sqrt{6}$，$|\vec{m}| =\sqrt{0^2+0^2+1^2} = 1$
$\therefore \cos\langle\vec{m}, \vec{n}\rangle = \frac{\vec{m} \cdot \vec{n}}{|\vec{m}||\vec{n}|} = \frac{2}{\sqrt{6} \times 1} = \frac{\sqrt{6}}{3}$
设平面 $ABC$ 与平面 $A_1BE$ 所成角为 $\theta$
$\therefore \cos\theta = |\cos\langle\vec{m}, \vec{n}\rangle| = \frac{\sqrt{6}}{3}$
$\therefore \sin\theta = \sqrt{1-\cos^2\theta} = \frac{\sqrt{3}}{3}$
答：正弦值为 $\frac{\sqrt{3}}{3}$
