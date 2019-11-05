# entropy_labelling
## confidential valueをどう選り分けるか?
<img src="https://latex.codecogs.com/gif.latex?P(Y=k|X=\boldsymbol{\textbf{x}})" />

1. 標準的な方法  
<img src="https://latex.codecogs.com/gif.latex?\arg\max_{k}P(Y=k|X=\boldsymbol{\mathbf{x}})" />をとってlabelとする

2. クラス数Kを使った素朴な方法  
<img src="https://latex.codecogs.com/gif.latex?\frac{1}{K}&space;\leq&space;P(Y=k|X=\boldsymbol{\mathbf{x}})" />を満たすものを全てlabelsとする
つまり1/K (一様分布のとき) をthrshldにしており，「どのクラスに属するのかまったく検討がつかない」という状態に相当する  
これを少しでも上回る&Rightarrow;「何も当てがないよりは，多少勘がある」ということ  
そうしたクラスを全てlabel化するというモデル  

3. エントロピーを用いた方法 (提案手法)  
エントロピーH(P)は，annの平均的な確信度の「低さ」を表す指標  
-H(P)は，annの平均的な確信度の「高さ」を表す指標

<img src="https://latex.codecogs.com/gif.latex?\frac{1}{K}&space;\leq&space;P(Y=k|X=\boldsymbol{\mathbf{x}})" />  
pkの自己情報量と，平均との比較  

H(P)を，「各kがlabelとして採用されるか」のthrshldに使っている  
    これが低い場合は，すなわち「偏りが大きいようなP(分布)をもつannは自信をもって判断している」ということを反映している  
    そうでければ初めから偏りなどつけないはず...
    
さらに直感的にするため，両辺negしてlogもはずす  

<img src="https://latex.codecogs.com/gif.latex?p_k&space;\geq&space;{2}^{-H(P)}" />  
すなわち，右辺を超えられる確信度pkをもつ全てのkを，ラベルとして採用する  

分布Pという全体の様相を考慮した動的thrshld  

