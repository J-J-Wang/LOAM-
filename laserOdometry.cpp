#include <cmath>

#include <loam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

const float scanPeriod = 0.1;

const int skipFrameNum = 1;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;

double timeLaserCloudFullRes = 0;
double timeImuTrans = 0;

bool newCornerPointsSharp = false;
bool newCornerPointsLessSharp = false;
bool newSurfPointsFlat = false;
bool newSurfPointsLessFlat = false;
bool newLaserCloudFullRes = false;
bool newImuTrans = false;
//接收前一节点  传过来的数据
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<pcl::PointXYZ>::Ptr imuTrans(new pcl::PointCloud<pcl::PointXYZ>());


pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());


pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<PointType>());

int laserCloudCornerLastNum;
int laserCloudSurfLastNum;

int pointSelCornerInd[40000];
float pointSearchCornerInd1[40000];
float pointSearchCornerInd2[40000];

int pointSelSurfInd[40000];
float pointSearchSurfInd1[40000];
float pointSearchSurfInd2[40000];
float pointSearchSurfInd3[40000];

float transform[6] = {0};
float transformSum[6] = {0};

float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollLast = 0, imuPitchLast = 0, imuYawLast = 0;
float imuShiftFromStartX = 0, imuShiftFromStartY = 0, imuShiftFromStartZ = 0;
float imuVeloFromStartX = 0, imuVeloFromStartY = 0, imuVeloFromStartZ = 0;
//TransformToStart() 当前点云中的点相对第一个点去除因匀速运动产生的畸变，
//效果相当于得到在点云扫描开始位置坐标系下  的扫描得到的点云
void TransformToStart(PointType const * const pi, PointType * const po)
{
  //插值系数计算，云中每个点的相对时间/点云周期（0.1）相当于*10
  float s = 10 * (pi->intensity - int(pi->intensity));
  //线性插值：根据每个点在点云中的相对位置关系，乘以相应的旋转平移系数
  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];
  //平移后绕z轴旋转（-rz）
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);
  //绕x轴旋转（-rx）
  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;
  //绕y轴旋转（-ry）
  po->x = cos(ry) * x2 - sin(ry) * z2;
  po->y = y2;
  po->z = sin(ry) * x2 + cos(ry) * z2;
  po->intensity = pi->intensity;
}
//TransformToEnd() 当前点云中的点相对第一个点去除因匀速运动产生的畸变，
//效果相当于得到在点云扫描结束位置坐标系下  的扫描得到的点云
void TransformToEnd(PointType const * const pi, PointType * const po)
{
  //插值系数计算
  float s = 10 * (pi->intensity - int(pi->intensity));

  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  float x3 = cos(ry) * x2 - sin(ry) * z2;
  float y3 = y2;
  float z3 = sin(ry) * x2 + cos(ry) * z2;

  rx = transform[0];
  ry = transform[1];
  rz = transform[2];
  tx = transform[3];
  ty = transform[4];
  tz = transform[5];

  float x4 = cos(ry) * x3 + sin(ry) * z3;
  float y4 = y3;
  float z4 = -sin(ry) * x3 + cos(ry) * z3;

  float x5 = x4;
  float y5 = cos(rx) * y4 - sin(rx) * z4;
  float z5 = sin(rx) * y4 + cos(rx) * z4;

  float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
  float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
  float z6 = z5 + tz;

  float x7 = cos(imuRollStart) * (x6 - imuShiftFromStartX) 
           - sin(imuRollStart) * (y6 - imuShiftFromStartY);
  float y7 = sin(imuRollStart) * (x6 - imuShiftFromStartX) 
           + cos(imuRollStart) * (y6 - imuShiftFromStartY);
  float z7 = z6 - imuShiftFromStartZ;

  float x8 = x7;
  float y8 = cos(imuPitchStart) * y7 - sin(imuPitchStart) * z7;
  float z8 = sin(imuPitchStart) * y7 + cos(imuPitchStart) * z7;

  float x9 = cos(imuYawStart) * x8 + sin(imuYawStart) * z8;
  float y9 = y8;
  float z9 = -sin(imuYawStart) * x8 + cos(imuYawStart) * z8;

  float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
  float y10 = y9;
  float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

  float x11 = x10;
  float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
  float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

  po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
  po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
  po->z = z11;
  po->intensity = int(pi->intensity);
}
// 利用IMU修正旋转量
void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                       float alx, float aly, float alz, float &acx, float &acy, float &acz)
{
  float sbcx = sin(bcx);
  float cbcx = cos(bcx);
  float sbcy = sin(bcy);
  float cbcy = cos(bcy);
  float sbcz = sin(bcz);
  float cbcz = cos(bcz);

  float sblx = sin(blx);
  float cblx = cos(blx);
  float sbly = sin(bly);
  float cbly = cos(bly);
  float sblz = sin(blz);
  float cblz = cos(blz);

  float salx = sin(alx);
  float calx = cos(alx);
  float saly = sin(aly);
  float caly = cos(aly);
  float salz = sin(alz);
  float calz = cos(alz);

  float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
            - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
            - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
            - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
            - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
  acx = -asin(srx);

  float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
  
  float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
               - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
               - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
               + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
               - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
               + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
               + calx*cblx*salz*sblz);
  float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
               - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
               + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
               + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
               + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
               - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
               - calx*calz*cblx*sblz);
  acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
}
//  AccumulateRotation() 相对于第一个点云即原点，积累旋转量
void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                        float &ox, float &oy, float &oz)
{
  float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
  ox = -asin(srx);

  float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
               + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
  float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
               - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
  oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

  float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
               + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
  float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
               - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
  oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}


//接收 点云 IMIU，进行主要数据 时间  点云 除去无效值的序号 新一帧的flag 赋值 
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharp2)
{
  timeCornerPointsSharp = cornerPointsSharp2->header.stamp.toSec();

  cornerPointsSharp->clear();
  pcl::fromROSMsg(*cornerPointsSharp2, *cornerPointsSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
  newCornerPointsSharp = true;
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharp2)
{
  timeCornerPointsLessSharp = cornerPointsLessSharp2->header.stamp.toSec();

  cornerPointsLessSharp->clear();
  pcl::fromROSMsg(*cornerPointsLessSharp2, *cornerPointsLessSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsLessSharp,*cornerPointsLessSharp, indices);
  newCornerPointsLessSharp = true;
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsFlat2)
{
  timeSurfPointsFlat = surfPointsFlat2->header.stamp.toSec();

  surfPointsFlat->clear();
  pcl::fromROSMsg(*surfPointsFlat2, *surfPointsFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsFlat,*surfPointsFlat, indices);
  newSurfPointsFlat = true;
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlat2)
{
  timeSurfPointsLessFlat = surfPointsLessFlat2->header.stamp.toSec();

  surfPointsLessFlat->clear();
  pcl::fromROSMsg(*surfPointsLessFlat2, *surfPointsLessFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsLessFlat,*surfPointsLessFlat, indices);
  newSurfPointsLessFlat = true;
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudFullRes,*laserCloudFullRes, indices);
  newLaserCloudFullRes = true;
}

void imuTransHandler(const sensor_msgs::PointCloud2ConstPtr& imuTrans2)
{
  timeImuTrans = imuTrans2->header.stamp.toSec();

  imuTrans->clear();
  pcl::fromROSMsg(*imuTrans2, *imuTrans);

  imuPitchStart = imuTrans->points[0].x;
  imuYawStart = imuTrans->points[0].y;
  imuRollStart = imuTrans->points[0].z;

  imuPitchLast = imuTrans->points[1].x;
  imuYawLast = imuTrans->points[1].y;
  imuRollLast = imuTrans->points[1].z;

  imuShiftFromStartX = imuTrans->points[2].x;
  imuShiftFromStartY = imuTrans->points[2].y;
  imuShiftFromStartZ = imuTrans->points[2].z;

  imuVeloFromStartX = imuTrans->points[3].x;
  imuVeloFromStartY = imuTrans->points[3].y;
  imuVeloFromStartZ = imuTrans->points[3].z;

  newImuTrans = true;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserOdometry");
  ros::NodeHandle nh;
 //接收第一部分处理过的点云 
  ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 2, laserCloudSharpHandler);
  ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2, laserCloudLessSharpHandler);
  ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 2, laserCloudFlatHandler);
  ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2, laserCloudLessFlatHandler);
  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_cloud_2", 2, laserCloudFullResHandler);
  //接收IMU消息 point  4
  ros::Subscriber subImuTrans = nh.subscribe<sensor_msgs::PointCloud2> ("/imu_trans", 5, imuTransHandler);
  //发布消息
  ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
  ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
  ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2> ("/velodyne_cloud_3", 2);
  ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);

  //设置里程计和TF的坐标系
  nav_msgs::Odometry laserOdometry;
  laserOdometry.header.frame_id = "/camera_init";
  laserOdometry.child_frame_id = "/laser_odom";

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform laserOdometryTrans;
  laserOdometryTrans.frame_id_ = "/camera_init";
  laserOdometryTrans.child_frame_id_ = "/laser_odom";


  std::vector<int> pointSearchInd;//搜索到的点序
  std::vector<float> pointSearchSqDis;//搜索到的点平方距离
  //特征点 对应点 映射点 误差；
  PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;
  //退化标志
  bool isDegenerate = false;
  //P矩阵，预测矩阵
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

  int frameCount = skipFrameNum;//1
  ros::Rate rate(100); 
  bool status = ros::ok();

  //进入循环
  while (status) {
    ros::spinOnce();
    //数据是否都接收 而且 时差小于0.005
    if (newCornerPointsSharp && newCornerPointsLessSharp && newSurfPointsFlat && 
        newSurfPointsLessFlat && newLaserCloudFullRes && newImuTrans &&
        fabs(timeCornerPointsSharp - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeCornerPointsLessSharp - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeSurfPointsFlat - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeLaserCloudFullRes - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeImuTrans - timeSurfPointsLessFlat) < 0.005) {

       //重置flag
      newCornerPointsSharp = false;
      newCornerPointsLessSharp = false;
      newSurfPointsFlat = false;
      newSurfPointsLessFlat = false;
      newLaserCloudFullRes = false;
      newImuTrans = false;

      if (!systemInited) {
        //初始化
        //将 新一轮特征 和上一轮的交换
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        //构建 kd树
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

        //将上一时刻数据（边缘+平面）从ROS转化为pcl，再发布出去
        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        //记住原点的翻滚角和俯仰角
        transformSum[0] += imuPitchStart;
        transformSum[2] += imuRollStart;

        systemInited = true;
        continue;
      }

      laserCloudOri->clear();
      coeffSel->clear();
      // T平移量的初值赋值为加减速的位移量，为其梯度下降的方向
      //（沿用上次转换的T（一个sweep匀速模型），同时在其基础上减去匀速运动位移，即只考虑加减速的位移量）
      transform[3] -= imuVeloFromStartX * scanPeriod;
      transform[4] -= imuVeloFromStartY * scanPeriod;
      transform[5] -= imuVeloFromStartZ * scanPeriod;

      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();
        for (int iterCount = 0; iterCount < 25; iterCount++) {
          //边缘特征处理
          for (int i = 0; i < cornerPointsSharpNum; i++) {
            TransformToStart(&cornerPointsSharp->points[i], &pointSel);

            if (iterCount % 5 == 0) {
              std::vector<int> indices;
              pcl::removeNaNFromPointCloud(*laserCloudCornerLast,*laserCloudCornerLast, indices);
               // 1、kd-tree查找一个最近距离点j，Ind为索引，SqDis为距离平方
              kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

              int closestPointInd = -1, minPointInd2 = -1;
              //在最邻近点j附近(向上下三条扫描线以内)找到次临近点l
              //再次提醒：velodyne是2度一线，scanID相邻并不代表线号相邻，相邻线度数相差2度，也即线号scanID相差2
              if (pointSearchSqDis[0] < 25) {
                closestPointInd = pointSearchInd[0];
                //提取最近点线号
                int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);

                float pointSqDis, minPointSqDis2 = 25;//距离平方阈值25
                // 2、寻找当前帧k点，在上一帧中最邻近点j附近的次临近点l
                  // 向scanID增大的方向查找
                for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {
                  if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                               (laserCloudCornerLast->points[j].x - pointSel.x) + 
                               (laserCloudCornerLast->points[j].y - pointSel.y) * 
                               (laserCloudCornerLast->points[j].y - pointSel.y) + 
                               (laserCloudCornerLast->points[j].z - pointSel.z) * 
                               (laserCloudCornerLast->points[j].z - pointSel.z);
                  //保存该点，并更新距离判断阈值
                  if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }
                 //向scanID减小的方向查找，向下三条线
                for (int j = closestPointInd - 1; j >= 0; j--) {
                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                               (laserCloudCornerLast->points[j].x - pointSel.x) + 
                               (laserCloudCornerLast->points[j].y - pointSel.y) * 
                               (laserCloudCornerLast->points[j].y - pointSel.y) + 
                               (laserCloudCornerLast->points[j].z - pointSel.z) * 
                               (laserCloudCornerLast->points[j].z - pointSel.z);
                  //比较距离阈值，如果比上一个更近，则保存该点，否则还是上一个点
                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }
              }
               //记住组成线的点序
              pointSearchCornerInd1[i] = closestPointInd;//kd-tree最近距离点j，-1表示未找到满足的点
              pointSearchCornerInd2[i] = minPointInd2;//另一个最近的l，-1表示未找到满足的点
            }
             // 3、计算点到线的距离，根据距离分配权重
            if (pointSearchCornerInd2[i] >= 0) {
              tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
              tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

              float x0 = pointSel.x;
              float y0 = pointSel.y;
              float z0 = pointSel.z;
              float x1 = tripod1.x;
              float y1 = tripod1.y;
              float z1 = tripod1.z;
              float x2 = tripod2.x;
              float y2 = tripod2.y;
              float z2 = tripod2.z;
              //向量OA = (x0 - x1, y0 - y1, z0 - z1), 
              //向量OB = (x0 - x2, y0 - y2, z0 - z2)，
              向量AB = （x1 - x2, y1 - y2, z1 - z2）
              //论文点到直线的距离分子的模：向量OA OB的向量积(即叉乘)为：
              //|  i      j      k  |
              //|x0-x1  y0-y1  z0-z1|
              //|x0-x2  y0-y2  z0-z2|
              float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                         * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                         + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                         * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                         + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                         * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
              
               //分母AB的模：两个最近距离点之间的距离
              float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
              //点到线的距离，d = |向量OA 叉乘 向量OB|/|AB| 
              float ld2 = a012 / l12;

               //AB方向的单位向量与OAB平面的单位法向量的向量积在各轴上的分量（d的方向）
              //x轴分量i
              float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;
              
               //y轴分量j
              float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

              //z轴分量k
              float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                       + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
               
              pointProj = pointSel;
              pointProj.x -= la * ld2;
              pointProj.y -= lb * ld2;
              pointProj.z -= lc * ld2;

              float s = 1;
              if (iterCount >= 5) {//5次迭代之后开始增加权重因素
                s = 1 - 1.8 * fabs(ld2);
              }

              coeff.x = s * la;
              coeff.y = s * lb;
              coeff.z = s * lc;
              coeff.intensity = s * ld2;

              if (s > 0.1 && ld2 != 0) {//只保留权重大的，也即距离比较小的点，同时也舍弃距离为零的
                laserCloudOri->push_back(cornerPointsSharp->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          }
          //平面特征处理
          for (int i = 0; i < surfPointsFlatNum; i++) {
            TransformToStart(&surfPointsFlat->points[i], &pointSel);
            //对本次接收到的平面点,从上次接收到的点云曲率比较小的点中找三点组成平面，一个使用kd-tree查找，
            //另外一个在同一线上查找满足要求的，第三个在不同线上查找满足要求的
            if (iterCount % 5 == 0) {
              //kd-tree最近点查找，在经过体素栅格滤波之后的平面点中查找，一般平面点太多，滤波后最近点查找数据量小
              kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

              int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
            
              if (pointSearchSqDis[0] < 25) {
                //查找上一帧最近点j
                closestPointInd = pointSearchInd[0];
                //查找上一帧最近点m，l
                int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);
                //向上查找
                float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;
                for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
                  if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                               (laserCloudSurfLast->points[j].x - pointSel.x) + 
                               (laserCloudSurfLast->points[j].y - pointSel.y) * 
                               (laserCloudSurfLast->points[j].y - pointSel.y) + 
                               (laserCloudSurfLast->points[j].z - pointSel.z) * 
                               (laserCloudSurfLast->points[j].z - pointSel.z);

                  if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) {
                     if (pointSqDis < minPointSqDis2) {
                       minPointSqDis2 = pointSqDis;
                       minPointInd2 = j;
                     }
                  } else {
                     if (pointSqDis < minPointSqDis3) {
                       minPointSqDis3 = pointSqDis;
                       minPointInd3 = j;
                     }
                  }
                }
                //向下查找
                for (int j = closestPointInd - 1; j >= 0; j--) {
                  if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                               (laserCloudSurfLast->points[j].x - pointSel.x) + 
                               (laserCloudSurfLast->points[j].y - pointSel.y) * 
                               (laserCloudSurfLast->points[j].y - pointSel.y) + 
                               (laserCloudSurfLast->points[j].z - pointSel.z) * 
                               (laserCloudSurfLast->points[j].z - pointSel.z);

                  if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  } else {
                    if (pointSqDis < minPointSqDis3) {
                      minPointSqDis3 = pointSqDis;
                      minPointInd3 = j;
                    }
                  }
                }
              }
              //保存找到的三个最近点  这里论文有两个点是在同一个scan环中，代码好像没体现，只要上下3条线都行
              pointSearchSurfInd1[i] = closestPointInd;
              pointSearchSurfInd2[i] = minPointInd2;
              pointSearchSurfInd3[i] = minPointInd3;
            }
              //计算点到平面距离  并保存相关系数  和上面的差不多
            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
              tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
              tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
              tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

              float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                       - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
              float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                       - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
              float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                       - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
              float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

              float ps = sqrt(pa * pa + pb * pb + pc * pc);
              pa /= ps;
              pb /= ps;
              pc /= ps;
              pd /= ps;

              float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

              pointProj = pointSel;
              pointProj.x -= pa * pd2;
              pointProj.y -= pb * pd2;
              pointProj.z -= pc * pd2;

              float s = 1;
              if (iterCount >= 5) {
                s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                  + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
              }

              coeff.x = s * pa;
              coeff.y = s * pb;
              coeff.z = s * pc;
              coeff.intensity = s * pd2;

              if (s > 0.1 && pd2 != 0) {
                laserCloudOri->push_back(surfPointsFlat->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          }

          int pointSelNum = laserCloudOri->points.size();
          if (pointSelNum < 10) {
            continue;
          }
           
          cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
          for (int i = 0; i < pointSelNum; i++) {
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

             // 1.计算matA,matB矩阵
            float s = 1;
            float srx = sin(s * transform[0]);
            float crx = cos(s * transform[0]);
            float sry = sin(s * transform[1]);
            float cry = cos(s * transform[1]);
            float srz = sin(s * transform[2]);
            float crz = cos(s * transform[2]);
            float tx = s * transform[3];
            float ty = s * transform[4];
            float tz = s * transform[5];
              // J对rx,ry,rz,tx,ty,tz求解偏导
            float arx = (-s*crx*sry*srz*pointOri.x + s*crx*crz*sry*pointOri.y + s*srx*sry*pointOri.z 
                      + s*tx*crx*sry*srz - s*ty*crx*crz*sry - s*tz*srx*sry) * coeff.x
                      + (s*srx*srz*pointOri.x - s*crz*srx*pointOri.y + s*crx*pointOri.z
                      + s*ty*crz*srx - s*tz*crx - s*tx*srx*srz) * coeff.y
                      + (s*crx*cry*srz*pointOri.x - s*crx*cry*crz*pointOri.y - s*cry*srx*pointOri.z
                      + s*tz*cry*srx + s*ty*crx*cry*crz - s*tx*crx*cry*srz) * coeff.z;

            float ary = ((-s*crz*sry - s*cry*srx*srz)*pointOri.x 
                      + (s*cry*crz*srx - s*sry*srz)*pointOri.y - s*crx*cry*pointOri.z 
                      + tx*(s*crz*sry + s*cry*srx*srz) + ty*(s*sry*srz - s*cry*crz*srx) 
                      + s*tz*crx*cry) * coeff.x
                      + ((s*cry*crz - s*srx*sry*srz)*pointOri.x 
                      + (s*cry*srz + s*crz*srx*sry)*pointOri.y - s*crx*sry*pointOri.z
                      + s*tz*crx*sry - ty*(s*cry*srz + s*crz*srx*sry) 
                      - tx*(s*cry*crz - s*srx*sry*srz)) * coeff.z;

            float arz = ((-s*cry*srz - s*crz*srx*sry)*pointOri.x + (s*cry*crz - s*srx*sry*srz)*pointOri.y
                      + tx*(s*cry*srz + s*crz*srx*sry) - ty*(s*cry*crz - s*srx*sry*srz)) * coeff.x
                      + (-s*crx*crz*pointOri.x - s*crx*srz*pointOri.y
                      + s*ty*crx*srz + s*tx*crx*crz) * coeff.y
                      + ((s*cry*crz*srx - s*sry*srz)*pointOri.x + (s*crz*sry + s*cry*srx*srz)*pointOri.y
                      + tx*(s*sry*srz - s*cry*crz*srx) - ty*(s*crz*sry + s*cry*srx*srz)) * coeff.z;

            float atx = -s*(cry*crz - srx*sry*srz) * coeff.x + s*crx*srz * coeff.y 
                      - s*(crz*sry + cry*srx*srz) * coeff.z;
  
            float aty = -s*(cry*srz + crz*srx*sry) * coeff.x - s*crx*crz * coeff.y 
                      - s*(sry*srz - cry*crz*srx) * coeff.z;
  
            float atz = s*crx*sry * coeff.x - s*srx * coeff.y - s*crx*cry * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
          }
           // 2.最小二乘法计算（QR分解）
          cv::transpose(matA, matAt);
          matAtA = matAt * matA;
          matAtB = matAt * matB;
           //求解matAtA * matX = matAtB
          cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

          // 3.出现退化用预测矩阵P
          if (iterCount == 0) {
            //特征值1*6矩阵
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
             //特征向量6*6矩阵
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
             //求解特征值/特征向量
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {10, 10, 10, 10, 10, 10};
            for (int i = 5; i >= 0; i--) {
              if (matE.at<float>(0, i) < eignThre[i]) {//特征值太小，则认为处在兼并环境中，发生了退化
                for (int j = 0; j < 6; j++) {//对应的特征向量置为0
                  matV2.at<float>(i, j) = 0;
                }
                isDegenerate = true;//退化flag 置为真
              } else {
                break;
              }
            }
             //计算P矩阵
            matP = matV.inv() * matV2;
          }

          if (isDegenerate) {//如果发生退化，只使用预测矩阵P计算
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
          }

          //------- (bug fix: sometime the L-M optimization result matX contains NaN, which will break the whole node)
          //判断是否非数字
          if (isnan(matX.at<float>(0, 0)) || isnan(matX.at<float>(1, 0)) || isnan(matX.at<float>(2, 0)) || isnan(matX.at<float>(3, 0)) || isnan(matX.at<float>(4, 0)) || isnan(matX.at<float>(5, 0)))
          {
            printf("[USER WARN]laser Odometry: NaN found in var \"matX\", this L-M optimization step is going to be ignored.\n");
          }
          else{
            //累加每次迭代的旋转平移量
            transform[0] += matX.at<float>(0, 0);
            transform[1] += matX.at<float>(1, 0);
            transform[2] += matX.at<float>(2, 0);
            transform[3] += matX.at<float>(3, 0);
            transform[4] += matX.at<float>(4, 0);
            transform[5] += matX.at<float>(5, 0);
          }
          //-------

          //计算旋转平移量，如果很小就停止迭代
          float deltaR = sqrt(
                              pow(rad2deg(matX.at<float>(0, 0)), 2) +
                              pow(rad2deg(matX.at<float>(1, 0)), 2) +
                              pow(rad2deg(matX.at<float>(2, 0)), 2));
          float deltaT = sqrt(
                              pow(matX.at<float>(3, 0) * 100, 2) +
                              pow(matX.at<float>(4, 0) * 100, 2) +
                              pow(matX.at<float>(5, 0) * 100, 2));

          if (deltaR < 0.1 && deltaT < 0.1) {//迭代终止条件
            break;
          }
        }
      }
// 1.局部坐标系转移到全局坐标系
      float rx, ry, rz, tx, ty, tz;
      // 1)旋转角累计变化量相对于原点rx,ry,rz
      //相对于第一个点云即原点，积累旋转量
      AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
                         -transform[0], -transform[1] * 1.05, -transform[2], rx, ry, rz);

      float x1 = cos(rz) * (transform[3] - imuShiftFromStartX) 
               - sin(rz) * (transform[4] - imuShiftFromStartY);
      float y1 = sin(rz) * (transform[3] - imuShiftFromStartX) 
               + cos(rz) * (transform[4] - imuShiftFromStartY);
      float z1 = transform[5] * 1.05 - imuShiftFromStartZ;

      float x2 = x1;
      float y2 = cos(rx) * y1 - sin(rx) * z1;
      float z2 = sin(rx) * y1 + cos(rx) * z1;
      // 2)平移量相对于原点tx,ty,tz
      tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
      ty = transformSum[4] - y2;
      tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);
      //根据IMU修正旋转量，插入imu旋转，更新位姿
      PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart, 
                        imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);
      // 得到世界坐标系下的转移矩阵T
      transformSum[0] = rx;
      transformSum[1] = ry;
      transformSum[2] = rz;
      transformSum[3] = tx;
      transformSum[4] = ty;
      transformSum[5] = tz;
      //欧拉角转换成四元数，再发布出去四元数和平移量
      geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(rz, -rx, -ry);

      laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometry.pose.pose.orientation.x = -geoQuat.y;
      laserOdometry.pose.pose.orientation.y = -geoQuat.z;
      laserOdometry.pose.pose.orientation.z = geoQuat.x;
      laserOdometry.pose.pose.orientation.w = geoQuat.w;
      laserOdometry.pose.pose.position.x = tx;
      laserOdometry.pose.pose.position.y = ty;
      laserOdometry.pose.pose.position.z = tz;
      pubLaserOdometry.publish(laserOdometry);

      laserOdometryTrans.stamp_ = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
      laserOdometryTrans.setOrigin(tf::Vector3(tx, ty, tz));
      tfBroadcaster.sendTransform(laserOdometryTrans);


//对k+1时刻的less特征点（边缘+平面）转移至k+1时刻的sweep的结束位置处的雷达坐标系下
      int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
      for (int i = 0; i < cornerPointsLessSharpNum; i++) {
        TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
      }

      int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
      for (int i = 0; i < surfPointsLessFlatNum; i++) {
        TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
      }
      // 点云全部点，每间隔一个点云数据相对点云最后一个点进行畸变校正
      frameCount++;
      if (frameCount >= skipFrameNum + 1) {
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
        }
      }

      pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
      cornerPointsLessSharp = laserCloudCornerLast;
      laserCloudCornerLast = laserCloudTemp;

      laserCloudTemp = surfPointsLessFlat;
      surfPointsLessFlat = laserCloudSurfLast;
      laserCloudSurfLast = laserCloudTemp;

      laserCloudCornerLastNum = laserCloudCornerLast->points.size();
      laserCloudSurfLastNum = laserCloudSurfLast->points.size();
      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

      }
      //按照跳帧数，发布边缘点，平面点以及全部点给laserMapping(每隔一帧发一次)
      if (frameCount >= skipFrameNum + 1) {
        frameCount = 0;

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "/camera";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);
      }
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
