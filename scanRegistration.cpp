
#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::sin;
using std::cos;
using std::atan2;

const double scanPeriod = 0.1;
const int systemDelay = 20;


int systemInitCount = 0;
bool systemInited = false;

//16线 激光雷达
const int N_SCANS = 16;

float cloudCurvature[40000];
int cloudSortInd[40000];
int cloudNeighborPicked[40000];
int cloudLabel[40000];

int imuPointerFront = 0;
int imuPointerLast = -1;
const int imuQueLength = 200;

float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};


ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

void ShiftToStartIMU(float pointTime)
{
  /*
  求当前点的位移相对于点云起始点的位移畸变，
  先计算全局坐标系下的然后再转换到IMU起始点的坐标系中。 
  首先计算畸变位移，再根据rpy的反向，绕y,x,z轴分别旋转。
  即可将位移畸变从世界坐标系转移到局部坐标系。
  rpy即roll,pitch,yaw分别是绕着z,x,y轴旋转，
  这里回去要反向旋转。*/
  //计算相对于第一个点由于加减速产生的畸变位移(全局坐标系下畸变位移量delta_Tg)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;
/********************************************************************************
  rpy：Rz(roll).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Tg
  transfrom from the global frame to the local frame
  *********************************************************************************/

  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;

  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

void VeloToStartIMU()
{
  //计算局部坐标系下点云中的点相对第一个开始点由于加减速产生的的速度畸变（增量）
  //计算相对于第一个点由于加减速产生的畸变速度(全局坐标系下畸变速度增量delta_Vg)
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;
/********************************************************************************
    rpy:Rz(roll).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

void TransformToStartIMU(PointType *p)
{
  /*去除点云加减速产生的位移畸变。 
  先rpy轴将点转换到世界坐标系然后再ypr由世界坐标系转换到IMU起始点坐标系，
  最后减去加减速造成的非匀速畸变的值。*/

  /********************************************************************************
    Ry*Rx*Rz*Pl, transform point to the global frame
  *********************************************************************************/
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;


/********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    transfrom global points to the local frame
  *********************************************************************************/
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

void AccumulateIMUShift()
{
  //积分速度与位移
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];
  // 绕RPY旋转转换得到世界坐标系下的加速度值
  //绕z轴旋转(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  //绕x轴旋转(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  //绕y轴旋转(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;
  //上一个imu点
  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  //上一个点到当前点所经历的时间，即计算imu测量周期
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  //要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义
  if (timeDiff < scanPeriod) {//（隐含从静止开始运动）
  //位移计算：X（t+1）= X（t）+v*dt+1/2*a*t^2;
  //速度计算：V(t+1)=V(t)+a*t；
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff 
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;

    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  //延时，保证传感器都正常工作后进行下一步，丢弃前systemDelay=20个点云数据帧
  if (!systemInited) {
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }
//记录每个scan 有曲率点的起始序号索引
  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);
  
  //获取时间戳
  double timeScanCur = laserCloudMsg->header.stamp.toSec();

  //点云格式转换，因为ROS中的点云和PCL的不一致
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

  //移除无效点
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);

  //获取点云数量
  int cloudSize = laserCloudIn.points.size();
  //计算起始的角度，顺时针为负，逆时针为正
  /*反正切函数 atan2() 和正切函数 tan() 的功能恰好相反：tan() 是已知一个角的弧度值，求该角的正切值；
  而 atan2() 是已知一个角的正切值（也就是 y/x），求该角的弧度值。
  atan2  以弧度表示的 y/x 的反正切的值，取值范围介于 -pi 到 pi 之间（不包括 -pi）
  而ATAN(a/b)的取值范围介于-pi/2到pi/2之间（不包括±pi/2)
  */

  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  // 扫描结束点的旋转角，加2*pi使点云旋转周期为2*pi
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;
  //结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描
  //正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);

  //给 每一个点赋予不同的行，也就是每一个环，scan或者ring

  for (int i = 0; i < cloudSize; i++) {
    //这里是坐标系转换   原本的激光雷达我看手册是  右手坐标系 ，这么一转换不知道在干吗？？？？
    //下面计算俯仰角，换成转换之前的，也是用右手坐标系好理解
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;
   //计算俯仰角从而推算 属于哪一行scan
    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;
    int scanID;
    //进行小数点的四舍五入
    int roundedAngle = int(angle + (angle<0.0?-0.5:+0.5)); 
    /*
    Laser ID    |    Vertical Angle
        0                  -15°
        1                   1°
        2                  -13°
        3                   3°
        4                  -11°
        5                   5°
        6                  -9°         
        7                   7°
        8                  -7°
        9                   9°
        10                 -5°
        11                  11°
        12                 -3°
        13                  13°
        14                 -1°
        15                  15°
    */
   //根据上表进行对应激光点所述的SCAN行号
    if (roundedAngle > 0){
      scanID = roundedAngle;
    }
    else {
      scanID = roundedAngle + (N_SCANS - 1);
    }
    //过滤点，只挑选[-15度，+15度]范围内的点,scanID属于[0,15]，剔除16线以外的杂点
    if (scanID > (N_SCANS - 1) || scanID < 0 ){
      count--;
      continue;
    }

   //计算该点的旋转角
    float ori = -atan2(point.x, point.z);
    //确保-pi/2 < ori - startOri < 3*pi/2
    if (!halfPassed) {
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }
    //判断是否过半
      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;
    //确保-3*pi/2 < ori - endOri < pi/2
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }
    //插补计算时间，并存在属性里，整数位为行号，小数位为起始点到当前点的时间差
    float relTime = (ori - startOri) / (endOri - startOri);
    point.intensity = scanID + scanPeriod * relTime;

    //IMU插值计算点云中点位置
    //这里不着急，待会看完其他几个函数再回来理一遍
    if (imuPointerLast >= 0) {
      //计算当前点相对于当前帧起始点的时间差
      //timeScanCur + pointTime为当前点的时间戳
      float pointTime = relTime * scanPeriod;
      //寻找是否有当前点的时间戳小于IMU的时间戳的IMU数据位置:imuPointerFront
      //保证IMU数据信息可用
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }
      //如果没找到,此时imuPointerFront==imuPointerLast,
      //只能以当前收到的最新的IMU的速度，位移，欧拉角作为当前点的速度，位移，欧拉角使用
      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else {
         //按时间距离进行线性插值
         //角度 RPY
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        //yaw取值范围 [-pi/2,pi/2] ? ?
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }
        //速度
        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;
        //位移
        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }
      //如果是第一个点,记住点云起始位置的速度，位移，欧拉角
      if (i == 0) {
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {
      //计算之后每个点相对于第一个点的由于加减速非匀速运动产生的位移、速度畸变，
      //并对点云中的每个点位置信息重新补偿矫正
      // Lidar位移、速度转移到IMU起始坐标系下
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }
    //将点保存到对应的行
    /* std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
    这里有些疑问，定义的形式如上，应该是初始值为16个元素的vector，可是到这里就变成了16个动态数组
    我通过右键查看定义，在original_src找到定义：
      00040 pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudScans[16];
    按照这个定义就是没错的，待查 STL ？？
    就当vector数组理解 
      */
    laserCloudScans[scanID].push_back(point);
  }
  //更新点云的数量，之前剔除了不是0-15行的一些点，所以数量会降低
  cloudSize = count;
  
  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  //将16个环的点云拼接，按照行号放入一个容器
 for (int i = 0; i < N_SCANS; i++) {
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  //计算去除 前五个点  后五个点的点云的曲率
  // 该点与周围10个点的偏差，参考论文公式
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;
    //存储相关序号信息
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
    cloudSortInd[i] = i;
    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;

    if (int(laserCloud->points[i].intensity) != scanCount) {
      scanCount = int(laserCloud->points[i].intensity);
    //曲率只取同一个行scan计算出来的，跨scan计算的曲率非法，排除，也即排除每个scan的前后五个点
      if (scanCount > 0 && scanCount < N_SCANS) {
        scanStartInd[scanCount] = i + 5;
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //第一行scan第一个点和最后一行scan的最后一个点的需要另外处理
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;
 
 //排除瑕疵点： 避免周围点已被选择从而保证特征点分布均匀，或者局部平行于激光束的局部平面上的点 和  被遮挡点正是
  for (int i = 5; i < cloudSize - 6; i++) {
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
    //先从第二种排除容易被斜面挡住的点以及离群点开始
    //具体看文章分析
    if (diff > 0.1) {

      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }
  //提出第一种瑕疵点
    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }

  //进行特征提取
  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  //对每一层激光点（16层）分割区域 6个 起始sp 终止ep
  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    for (int j = 0; j < 6; j++) {
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      // 将曲率排序--从小到大  冒泡排序法
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }
      //挑选每个分段的曲率很大和比较大的点，作为边缘特征
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];
        //曲率比较大，并且未被筛选
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > 0.1) {
        
          largestPickedNum++;
          //挑选曲率最大的前2个点放入sharp点集合
          if (largestPickedNum <= 2) {
            cloudLabel[ind] = 2; //2代表点曲率很大，最优
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) { //挑选曲率最大的前20个点放入less sharp点集合
            cloudLabel[ind] = 1;//1代表点曲率比较尖锐，次优
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;//20个点以上不要，每个区域只取2+20个
          }
          //将点标记为已选择
          cloudNeighborPicked[ind] = 1;
           //将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，
           //防止特征点聚集，使得特征点在每个方向上尽量分布均匀
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //同理操作 选取曲率很小和比较小的点，作为平面特征
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];
        ////如果曲率的确比较小，并且未被筛选出
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//-1代表曲率很小的点
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;//只选最小的四个，剩下的Label==0,就都是曲率比较小的
          if (smallestPickedNum >= 4) {
            break;
          }
          //同样防止特征点聚集
          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }
      //最后把剩余点（包括之前被排除的点）全部归入平面点中less flat类别中
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }
    //由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);
     //less flat点汇总
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //借助 ros消息 不同的点云消息包括： 原始 最优边缘  次优边缘 最优平面 次优平面 发布出去
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);


  //发布IMU 
  //publich IMU消息,由于循环到了最后，
  //因此是Cur都是代表最后一个点，即最后一个点的欧拉角，
  //畸变位移及一个点云周期增加的速度
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  //起始点欧拉角
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;
 //最后一个点的欧拉角
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;
 //最后一个点相对于第一个点的畸变位移和速度
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  // 通过四元数获得roll,pitch,yaw
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
  // 去除重力加速度的影响并进行坐标轴交换，统一到z轴向前,x轴向左的右手坐标系
  // 交换过后RPY对应fixed axes ZXY(RPY---ZXY)。Now R = Ry(yaw)*Rx(pitch)*Rz(roll).
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //循环移位效果，形成环形数组
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;
 //位姿估算 
  AccumulateIMUShift();
}

int main(int argc, char** argv)
{
  //ros::init(argc, argv, "scanRegistration");
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

//订阅 激光 和 惯性传感器的消息
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/velodyne_points", 2, laserCloudHandler);

  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);


//发布处理的点云
  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/velodyne_cloud_2", 2);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);

  ros::spin();

  return 0;
}

