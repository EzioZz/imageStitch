#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;


 
//重要参数
const int NUM=21;//输入图片的张数
const double lower_k = 0.0005; /// lower_E = lower_k * max_E
const double k = 0.05;// det - k * trace*trace
const int threshold1 = 60;//22
const double threshold2 = 2;
//if(dis1[i][0] < threshold1 && dis2[j][1] == i && (dis1[i][2] / dis1[i][1]) > threshold2){


//所用到的数据结构
struct FeaturePoint{ //每一个特征点的结构体
    cv::Point2i P;
    vector<int> vec; //brief描述子的特征点
    FeaturePoint(){}
    FeaturePoint(cv::Point2i _P){
        P = _P;
    }
};

struct Pair{  //在每一个特征点上取pattern时用到的点对
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

struct Corners{  //每一张图在世界坐标系下的特征点
    cv::Point2i top_left;
    cv::Point2i top_right;
    cv::Point2i bottom_left;
    cv::Point2i bottom_right;
};

vector<double> Corners_x;
vector<double> Corners_y;
double trans_x; //为了将图像显示在正数范围内需要平移的距离
double trans_y; //trans_x = -min_x; trans_y = -min_y;



//产生的数
cv::Mat stitchData[NUM];//最后用来拼接生成数据
cv::Mat oriGrey[NUM];
cv::Mat oriRGB[NUM];
vector<FeaturePoint> FeaturePoints[NUM]; //每一张图都有组由harris获得的特征点
double Emap[1921][1081];
vector<cv::Point2i> matchPoints[2];


//主要函数
void inputData();//读入数据
void getFeaturePoints();//用harris计算特征点
void NMS(int NO, double Emax, int blockSize);//非极大值抑制
void getDescriptors();//计算berif描述子
double hammingDistance(vector<int> vec1, vector<int> vec2); //两个向量间的汉明距离
void findMatchPoints(int no1, int no2, vector<cv::Point2i> &goodmatch1, vector<cv::Point2i> &goodmatch2); //进行描述子的匹配
void stitch(int stepSize);//
cv::Mat myfindHomography(vector<cv::Point2i> goodmatch1, vector<cv::Point2i> goodmatch2);

int main(){
    inputData();
    getFeaturePoints();
    getDescriptors();
    stitch(2);//2表示步长
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
        oriRGB[i].copyTo(stitchData[i]);
        cv::GaussianBlur(oriGrey[i], oriGrey[i], cv::Size(3,3), 0, 0, 4);
        // imshow("i",oriGrey[i]);
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
        NMS(no, Emax, 10);
        cout<<FeaturePoints[no].size()<<endl;
        for(auto var : FeaturePoints[no]){
            cv::circle(oriRGB[no], var.P,3,cv::Scalar(0,0,255),1);
        }
        // 可视化，获取特征点之后的图片
        // cv::imshow(to_string(no), oriRGB[no]);
        // cv::waitKey(0);
    }

}





void NMS(int NO, double Emax, int blockSize){
    /** @brief 非极大值抑制
     * 
     */
    blockSize/=2;
    for(int i=51; i<oriGrey[0].rows-51; i++){
        for(int j=51; j<oriGrey[0].cols-51; j++){
            int isLocalMax = 1;
            if(Emap[i][j] > lower_k * Emax){
                for(int ii = i-blockSize; ii<i+blockSize; ii++){
                    for(int jj = j-blockSize; jj<j+blockSize; jj++){
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

void findMatchPoints(int no1, int no2, vector<cv::Point2f> &goodmatch1, vector<cv::Point2f> &goodmatch2){
    // @brief 给定两张图的序号在两张图的harris响应中 进行描述子的匹配
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

    //描述子匹配可视化部分
    //将匹配点之间画上一条线
    //创建连接后存入的图像，两幅图像按左右排列，所以列数+1
    // cv::Mat result(oriRGB[no1].rows,oriRGB[no1].cols+oriRGB[no2].cols+1,oriRGB[no1].type());
    // oriRGB[no1].colRange(0,oriRGB[no1].cols).copyTo(result.colRange(0,oriRGB[no1].cols));
    // oriRGB[no2].colRange(0,oriRGB[no2].cols).copyTo(result.colRange(oriRGB[no1].cols+1,result.cols));
    // for(int i=0; i<goodmatch1.size(); i++){
    //     //将匹配点之间画上一条线
    //     cv::line(result, goodmatch1[i], cv::Point2i(goodmatch2[i].x + oriRGB[no1].cols, goodmatch2[i].y), cv::Scalar(0,0,255),1);
    // }
}

cv::Mat myfindHomography(vector<cv::Point2i> goodmatch1, vector<cv::Point2i> goodmatch2){
    
}

void CalCorners(cv::Mat src, cv::Mat H){
    Corners c;
    // Mat array = (Mat_<double>(3, 3) << 0, -1, 5, -1, 5, -1, 0, -1, 0);
    cv::Mat top_right = (cv::Mat_<double>(3,1)   << 1.0 * src.cols, 1.0 * src.rows, 1.0 );
    cv::Mat top_left = (cv::Mat_<double>(3,1)    << 0, 1.0 * src.rows, 1.0);
    cv::Mat bottom_right = (cv::Mat_<double>(3,1)<< 1.0 * src.cols, 0, 1.0);
    cv::Mat bottom_left = (cv::Mat_<double>(3,1) << 0, 0, 1.0);
    cv::Mat H_top_right = H * top_right;
    cv::Mat H_top_left = H * top_left;
    cv::Mat H_bottom_right = H * bottom_right;
    cv::Mat H_bottom_left = H * bottom_left;
    c.top_right = cv::Point2i( H_top_right.at<double>(0,0), H_top_right.at<double>(1,0));
    c.top_left = cv::Point2i( H_top_left.at<double>(0,0), H_top_left.at<double>(1,0));
    c.bottom_right = cv::Point2i( H_bottom_right.at<double>(0,0), H_bottom_right.at<double>(1,0));
    c.bottom_left = cv::Point2i( H_bottom_left.at<double>(0,0), H_bottom_left.at<double>(1,0));
    // cout<<c.top_left<<","<<c.top_right<<","<<c.bottom_left<<","<<c.bottom_right<<endl;
    Corners_x.push_back(c.top_left.x); Corners_x.push_back(c.top_right.x); Corners_x.push_back(c.bottom_left.x); Corners_x.push_back(c.bottom_right.x);
    Corners_y.push_back(c.top_left.y); Corners_y.push_back(c.top_right.y); Corners_y.push_back(c.bottom_left.y); Corners_y.push_back(c.bottom_right.y); 
}

cv::Scalar interpolatio(double x, double y, int no){  
    //双线性插值，返回插值点的RGB
    cv::Point2i P0((int)x, (int)y);   
    cv::Point2i P1((int)x, (int)y + 1);
    cv::Point2i P2((int)x + 1, (int)y + 1);
    cv::Point2i P3((int)x + 1, (int)y);
    cv::Scalar num0 = 1.0 * stitchData[no].at<cv::Vec3b>(P0.x, P0.y);
    cv::Scalar num1 = 1.0 * stitchData[no].at<cv::Vec3b>(P1.x, P1.y);
    cv::Scalar num3 = 1.0 * stitchData[no].at<cv::Vec3b>(P3.x, P3.y);
    cv::Scalar num2 = 1.0 * stitchData[no].at<cv::Vec3b>(P2.x, P2.y);
    cv::Scalar dnumdy1 = 1.0 * (num1-num0);
    cv::Scalar dnumdy2 = 1.0 * (num2-num3);
    cv::Scalar num4 = num0 + ((1.0)*y - (1.0)*P0.y)*dnumdy1;
    cv::Scalar num5 = num3 + ((1.0)*y - (1.0)*P3.y)*dnumdy2;

    cv::Point2i P4; cv::Point2i P5;
    P4.x = P1.x;
    P5.x = P2.x;
    cv::Scalar dnumdx = 1.0*(num5-num4)/1.0;
    cv::Scalar numpt = num4 + dnumdx * ((1.0)*x - (1.0)*P4.x);
    return numpt;
}



void stitch(int stepSize){
    // int no0 = 0;
    // int no1 = 2;
    trans_x = 0;
    trans_y = 0;
    Corners_x.clear();
    Corners_y.clear();
    Corners_x.push_back(0); Corners_x.push_back(oriRGB[0].cols);
    Corners_y.push_back(0); Corners_y.push_back(oriRGB[0].rows);
    cv::Mat Hs[NUM];//图片0到每一张图片的变换
    for(int i=0; i<=18; i+=2){ // i = 0,2,4,6,8,...
        int no0 = i; int no1 = i + 2;
        vector<cv::Point2f> goodmatch1;
        vector<cv::Point2f> goodmatch2;
        findMatchPoints(no0, no1, goodmatch1, goodmatch2);  //计算图片no0和no1的goodmatchPoints
        cv::Mat H = cv::findHomography(goodmatch1, goodmatch2, CV_RANSAC);//找到投影变换矩阵，从图像1到图像2的映射变换
        int H_no = i/2;
        if(H_no >= 1){
            Hs[H_no] = H * Hs[H_no-1];
        }
        else Hs[H_no] = H;
        cout<<i<<endl;
        CalCorners(oriRGB[no1], Hs[H_no].inv()); //将新图像变化到0号图坐标系后的点，加入Corners_x和Corners_y中
    }

    auto max_x_p = max_element(Corners_x.begin(), Corners_x.end());
    auto min_x_p = min_element(Corners_x.begin(), Corners_x.end());
    auto max_y_p = max_element(Corners_y.begin(), Corners_y.end());
    auto min_y_p = min_element(Corners_y.begin(), Corners_y.end());
    cout<<*max_x_p<<","<<*max_y_p<<","<<*min_x_p<<","<<*min_y_p<<endl;
    trans_x -= *min_x_p;
    trans_y -= *min_y_p;

    double maxHeight = *max_y_p - *min_y_p;
    double maxWidth = *max_x_p - *min_x_p; 
    cv::Mat res(maxHeight, maxWidth, CV_8UC3, cv::Scalar(0,0,0));

    // cv::Mat top_right = (cv::Mat_<double>(3,1) << 1.0 * src.cols, 1.0 * src.rows, 1.0 );
    for(int j = *min_y_p; j < *max_y_p; j++){
        for(int i = *min_x_p ; i < *max_x_p; i++){
            for(int no = 0; no<=20; no+=2){
                if(no == 0){
                    cv::Scalar temp = 1.0 * stitchData[0].at<cv::Vec3b>(cv::Point2i(i, j));
                    if(i<0 || j<0 || i>oriRGB[0].cols || j>oriRGB[0].rows) continue;
                    else{
                        res.at<cv::Vec3b>(cv::Point2i(i+trans_x, j+trans_y)) = cv::Vec3b(temp[0], temp[1], temp[2]);
                        no = 99999;
                    }
                }
                else{
                    cv::Mat p = (cv::Mat_<double>(3,1) << 1.0 * i, 1.0 * j, 1.0 );
                    int picture_no = no;
                    int H_no = no/2 - 1;
                    cv::Mat hp = Hs[H_no] * p;
                    int x = hp.at<double>(0,0);
                    int y = hp.at<double>(1,0);
                    if(x<0 || y<0 || x > oriRGB[picture_no].cols || y > oriRGB[picture_no].rows) continue;
                    else{
                        cv::Scalar temp = interpolatio(y, x, picture_no);
                        res.at<cv::Vec3b>(cv::Point2i(i+trans_x, j+trans_y)) = cv::Vec3b(temp[0], temp[1], temp[2]);
                        no = 99999; // 不再循环no下去
                    }
                }  
            }
        }
    }

    imshow("res", res);
    imwrite("../out/res.jpg", res);
    cv::waitKey(0);
}



//统计机器学习
//goodfellow
//pattern classfication




//https://zhuanlan.zhihu.com/p/36382429
//https://blog.csdn.net/hhyh612/article/details/79189983


/*
            if(i>=0 && j>=0) {
                cv::Scalar temp = 1.0 * oriRGB[no0].at<cv::Vec3b>(cv::Point2i(i, j));
                // cv::Scalar num2 = 1.0 * oriRGB[no].at<cv::Vec3b>(P2.x, P2.y);
                res.at<cv::Vec3b>(cv::Point2i(i+trans_x, j+trans_y)) = cv::Vec3b(temp[0], temp[1], temp[2]);
            }
            else{
                cv::Mat p = (cv::Mat_<double>(3,1) << 1.0 * i, 1.0 * j, 1.0 );
                p = H * p;
                int x = p.at<double>(0,0);
                int y = p.at<double>(1,0);
                if(x<0 || y<0 || x > oriRGB[no1].cols || y > oriRGB[no1].rows) continue;
                cv::Scalar temp = interpolatio( y, x, no1);
                // cout<<i+trans_x<<","<<j+trans_y<<endl;
                res.at<cv::Vec3b>(cv::Point2i(i+trans_x, j+trans_y)) = cv::Vec3b(temp[0], temp[1], temp[2]);
            }

*/