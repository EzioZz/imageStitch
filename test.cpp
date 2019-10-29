
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
 
//-----------------------------------【命名空间声明部分】---------------------------------------  
//      描述：包含程序所使用的命名空间  
//-----------------------------------------------------------------------------------------------   
using namespace cv;  
 
int main()  {  
	//以灰度模式载入图像并显示
	Mat srcImage = imread("/Users/yanyucheng/OneDrive/codeProjects/imageStitching/dataset/0019.bmp", 0);  
	imshow("原始图", srcImage);  
 
	//进行Harris角点检测找出角点
	Mat cornerStrength;  
	cornerHarris(srcImage, cornerStrength, 2, 3, 0.05);
 
	//对灰度图进行阈值操作，得到二值图并显示  
	Mat harrisCorner;  
	threshold(cornerStrength, harrisCorner, 0.001, 255, THRESH_BINARY);  
	imshow("角点检测后的二值效果图", harrisCorner);  
 
	waitKey(0);  
	return 0;  
} 
