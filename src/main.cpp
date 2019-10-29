#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;


 
//重要参数
const int NUM=21;//输入图片的张数
const double lower_k = 0.01; /// lower_E = lower_k * max_E
const double k = 0.05;// det - k * trace*trace
const int threshold1 = 40;
const double threshold2 = 2;
//if(dis1[i][0] < 40 && dis2[j][1] == i && (dis1[i][2] / dis1[i][1]) > 2){


//
struct FeaturePoint{
    cv::Point2i P;
    vector<int> vec; //brief描述子的特征点
    FeaturePoint(){}
    FeaturePoint(cv::Point2i _P){
        P = _P;
    }
};

struct Pair{
    int dx1;
    int dy1;
    int dx2;
    int dy2;
    Pair(){}
    Pair(int _dx1, int _dy1, int _dx2, int _dy2){
        dx1 = _dx1;
        dy1 = _dy1;
        dx2 = _dx2;
        dy2 = _dy2;
    }
};



//产生的数据
cv::Mat oriGrey[NUM];
cv::Mat oriRGB[NUM];
vector<FeaturePoint> FeaturePoints[NUM]; //每一张图都有组由harris获得的特征点
double Emap[1921][1081];
vector<cv::Point2i> matchPoints[2];



void inputData();
void getFeaturePoints();
void NMS(int NO, double Emax);
void getDescriptors();
double hammingDistance(vector<int> vec1, vector<int> vec2); //两个向量间的汉明距离
void findMatchPoints(int no1, int no2, vector<cv::Point2i> &goodmatch1, vector<cv::Point2i> &goodmatch2);
void stitch();

void test(){
    // vector<int> vec1,vec2;
    // vec1.push_back(1);
    // vec1.push_back(0);
    // vec2.push_back(0);
    // vec2.push_back(0);
    // cout<<hammingDistance(vec1, vec2);
    // cv::waitKey(0);
}

int main(){
    inputData();
    getFeaturePoints();
    getDescriptors();
    stitch();

}

void inputData(){
    for(int i=0; i<NUM; i++){
        string temp;
        if(i<10){
            temp = "/Users/yanyucheng/OneDrive/codeProjects/imageStitching/dataset/000" + to_string(i) + ".bmp";
        }
        else if(i<100){
            temp = "/Users/yanyucheng/OneDrive/codeProjects/imageStitching/dataset/00" + to_string(i) + ".bmp";
        }
        else cout<<"图片张数超过上限，需修改inputData()处代码"<<endl;
        oriGrey[i] = cv::imread(temp, 0);
        oriRGB[i] = cv::imread(temp);
        cv::GaussianBlur(oriGrey[i], oriGrey[i], cv::Size(3,3), 0, 0, 4);
        // imshow("i",oriGrey[i]);
        // cv::waitKey(0);
        // cv::circle(oriRGB[i], cv::Point2i(50, 50),4,cv::Scalar(0,0,255),3);
    }

}


void getFeaturePoints(){
    //使用harris来进行特征点提取
    //我观察输入数据好像没有很严重的尺度不变性
    cv::Mat fx = (cv::Mat_<double>(1,3) << -1.0, 0, 1.0);
    cv::Mat fy = (cv::Mat_<double>(3,1) << -1.0, 0, 1.0);
    cv::Mat Ix[NUM];    cv::Mat Iy[NUM];
    cv::Mat Ix2[NUM];   cv::Mat Iy2[NUM];
    cv::Mat Ixy[NUM];
    for(int i=0; i<NUM; i++){
        cv::filter2D(oriGrey[i], Ix[i], CV_64F, fx);
        cv::filter2D(oriGrey[i], Iy[i], CV_64F, fy);
        Ix2[i] = Ix[i].mul(Ix[i]);
        Iy2[i] = Iy[i].mul(Iy[i]);
        Ixy[i] = Iy[i].mul(Ix[i]);
    }
    memset(Emap,0,sizeof(Emap));
    for(int no = 0; no<NUM; no++){
        double Emax = -9999999;//用来存能量的最大值
        for(int i=51; i<oriGrey[no].rows-51; i++){
            for(int j=51; j<oriGrey[no].cols-51; j++){
                Eigen::Matrix<double, 2, 2> Matrix_22;
                //自相关矩阵
                double xx=0, yy=0, xy=0;
                for(int ii = i - 1; ii <= i + 1; ii++){
                    for(int jj = j-1; jj <= j + 1; jj++){
                        xx += Ix2[no].at<double>(ii,jj);
                        yy += Iy2[no].at<double>(ii,jj);
                        xy += Ixy[no].at<double>(ii,jj);
                    }
                }
                Matrix_22<< xx, xy, xy, yy;
                Matrix_22 /= 9;

                double E = Matrix_22.determinant() - k * Matrix_22.trace()*Matrix_22.trace();
                Emap[i][j] = E;
                Emax = max(E,Emax);
            }
        }
        NMS(no, Emax);
        cout<<FeaturePoints[no].size()<<endl;
        for(auto var : FeaturePoints[no]){
            cv::circle(oriRGB[no], var.P,3,cv::Scalar(0,0,255),1);
        }
        //可视化，获取特征点之后的图片
        // cv::imshow(to_string(no), oriRGB[no]);
        // cv::waitKey(0);
    }

}

void NMS(int NO, double Emax){
    //@berief 非极大值抑制
    for(int i=51; i<oriGrey[0].rows-51; i++){
        for(int j=51; j<oriGrey[0].cols-51; j++){
            int isLocalMax = 1;
            if(Emap[i][j] > lower_k * Emax){
                for(int ii = i-1; ii<i+1; ii++){
                    for(int jj = j-1; jj<j+1; jj++){
                        if(Emap[i][j] < Emap[ii][jj]){
                            isLocalMax = 0;
                        }
                    }
                }
                if(isLocalMax){
                    FeaturePoints[NO].push_back(FeaturePoint(cv::Point2i(j, i)));
                }
            }
        }
    }
}

void getDescriptors(){
    vector<Pair> pattern;
    for(int i=0; i<256; i++){
        pattern.push_back(Pair(rand()%51-25, rand()%51-25, rand()%51-25, rand()%51-25));
    }
    for(int no = 0; no < NUM; no++){//遍历每一张图像
        for(auto &var : FeaturePoints[no]){//遍历每一个特征点
            int x = var.P.y;
            int y = var.P.x;
            for(auto var_p : pattern){ //遍历128个特征
                int x1 = x + var_p.dx1;
                int y1 = y + var_p.dy1;
                int x2 = x + var_p.dx2;
                int y2 = y + var_p.dy2;
                //如果灰度值P1>P2则向量push_back 1，如果想等的则向量push_back 0, 反之则push_back 0
                if(oriGrey[no].at<uchar>(x1,y1) > oriGrey[no].at<uchar>(x2,y2)){
                    var.vec.push_back(1);
                }
                else{
                    var.vec.push_back(0);
                }
            }
        }
    }

}

double hammingDistance(vector<int> vec1, vector<int> vec2){
    int res = 0;
    for(int i=0; i<vec1.size(); i++){
        if(vec1[i] != vec2[i]){
            res += 1;
        }
    }
    return res;
}

void findMatchPoints(int no1, int no2, vector<cv::Point2i> &goodmatch1, vector<cv::Point2i> &goodmatch2){

    double dis1[FeaturePoints[no1].size()][4]; //0,1,2,3 最小距离，序号，次小距离，序号
    double dis2[FeaturePoints[no2].size()][4];

    for(int i=0; i<FeaturePoints[no1].size(); i++){
        for(int j=0; j<4; j++){
            dis1[i][j] = 9999;
        }
    }
    for(int i=0; i<FeaturePoints[no2].size(); i++){
        for(int j=0; j<4; j++){
            dis2[i][j] = 9999;
        }
    }

    for(int i=0; i<FeaturePoints[no1].size(); i++){
        for(int j=0; j<FeaturePoints[no2].size(); j++){
            double dis = hammingDistance(FeaturePoints[no1][i].vec, FeaturePoints[no2][j].vec);
            // cout<<dis<<endl;
            if(dis1[i][0] > dis){
                dis1[i][3] = dis1[i][1];
                dis1[i][2] = dis1[i][0];
                dis1[i][1] = j;
                dis1[i][0] = dis;
            }
            else if(dis1[i][2] > dis){
                dis1[i][3] = j;
                dis1[i][2] = dis;
            }

            if(dis2[j][0] > dis){
                dis2[j][3] = dis2[j][1];
                dis2[j][2] = dis2[j][0];
                dis2[j][1] = i;
                dis2[j][0] = dis;
            }
            else if(dis2[j][2] > dis){
                dis2[j][3] = i;
                dis2[j][2] = dis;
            }
        }
    }

    for(int i=0; i<FeaturePoints[no1].size(); i++){
        int j = dis1[i][1];
        if(dis1[i][0] < threshold1 && dis2[j][1] == i && ((1.0)*dis1[i][0] / (1.0)*dis1[i][2]) > threshold2){
            goodmatch1.push_back(FeaturePoints[no1][i].P);
            goodmatch2.push_back(FeaturePoints[no2][j].P);
        }
    }

    //将匹配点之间画上一条线
    //创建连接后存入的图像，两幅图像按左右排列，所以列数+1
    cv::Mat result(oriRGB[no1].rows,oriRGB[no1].cols+oriRGB[no2].cols+1,oriRGB[no1].type());
    oriRGB[no1].colRange(0,oriRGB[no1].cols).copyTo(result.colRange(0,oriRGB[no1].cols));
    oriRGB[no2].colRange(0,oriRGB[no2].cols).copyTo(result.colRange(oriRGB[no1].cols+1,result.cols));
    // cv::imshow("drawMatched", result);
    // cv::waitKey(0);

    cout<<"size1:"<<goodmatch1.size()<<endl;
    cout<<"size2:"<<goodmatch2.size()<<endl;
    for(int i=0; i<goodmatch1.size(); i++){
        //将匹配点之间画上一条线
        cv::line(result, goodmatch1[i], cv::Point2i(goodmatch2[i].x + oriRGB[no1].cols, goodmatch2[i].y), cv::Scalar(0,0,255),1);
    }
    cv::imshow(to_string(no1)+"_to_"+to_string(no2), result);
    cv::waitKey(0);

}

void stitch(){

    int mask[21]={};
    for(int i=0; i<NUM; i++){ 
        vector<cv::Point2i> goodmatch1;
        vector<cv::Point2i> goodmatch2;
        findMatchPoints(1, i, goodmatch1, goodmatch2);
        // cv::Mat H = findHomography(goodmatch1, goodmatch1);//找到投影变换矩阵


    }

}






//https://zhuanlan.zhihu.com/p/36382429
//https://blog.csdn.net/hhyh612/article/details/79189983
