# answerSheetScan
因为要当助教，批改答题卡，很无聊，想着能不能做一个识别答题卡的py程序，看了下git上的不少程序没法很符合自己的需求吧，参考了一些别人对答题卡的识别方法，采用的还是python上的opencv写了一个小原型（吐槽一下surface3的垃圾摄像头，不能对焦，设备有限，只能拿手机摄像头作摄像头）。

能用是能用了，不过识别率感人:cry:，填涂和线框重合的时候就没法识别了，这个问题研究了下可以通过识别凸域进行切割，有需要再改改吧，凑活自己能用:slightly_smiling_face:算是做个记录吧，算是菜鸟入门做了一点实际能用的东西。img里面放了一些测试图片。

依赖库:

1.numpy

2.pythonopencv

3.imutils

测试图:![image](https://github.com/realbad/answerSheetsScan/blob/master/img/ques.jpg)