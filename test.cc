#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <memory>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
using namespace std;
using namespace cv;

Mat total_img;

struct TreeNode{
    vector<KeyPoint> keypoints;
    int iniX, finalX, iniY, finalY;
    TreeNode(){}
    TreeNode(vector<KeyPoint> keypoints, int iniX, int finalX, int iniY, int finalY):keypoints(std::move(keypoints)), iniX(iniX), finalX(finalX), 
        iniY(iniY), finalY(finalY){
    }

    vector<TreeNode> divide_nodes(){
        vector<TreeNode> result(4);
        int middleX = (iniX + finalX)/2, middleY = (iniY + finalY)/2;
        vector<KeyPoint> keypoints_set[4];
        for(auto &keypoint:keypoints){
            if(keypoint.pt.x < middleX && keypoint.pt.y < middleY)
                keypoints_set[0].push_back(keypoint);
            else if(keypoint.pt.x < middleX && keypoint.pt.y > middleY)
                keypoints_set[1].push_back(keypoint);
            else if(keypoint.pt.x > middleX && keypoint.pt.y < middleY)
                keypoints_set[2].push_back(keypoint);
            else if(keypoint.pt.x > middleX && keypoint.pt.y > middleY)
                keypoints_set[3].push_back(keypoint);
        }
        result[0] = std::move(TreeNode(keypoints_set[0], iniX, middleX, iniY, middleY));
        result[1] = std::move(TreeNode(keypoints_set[1], iniX, middleX, middleY, finalY));
        result[2] = std::move(TreeNode(keypoints_set[2], middleX, finalX, iniY, middleY));
        result[3] = std::move(TreeNode(keypoints_set[3], middleX, finalX, middleY, finalY));
        return std::move(result);
    }
};

class ORBextractor{
public:
    int nlevels;
    double scale = 1.2;
    vector<double> mvScale;
    vector<double> mvInvScale;
    vector<Mat> img_pyramid;
    int iniThred = 20, nextThred = 7;
    double unitNumber;
    int W = 30;
    int N = 1000;
    using listIter = list<TreeNode>::iterator;
    struct ptr2nkp{
        listIter it;
        int nkp;
        ptr2nkp(listIter it, int nkp):it(it), nkp(nkp){}
    };

    ORBextractor(int nlevels):nlevels(nlevels){
        for(int i = 0; i < nlevels; i++){
            mvScale.emplace_back(pow(scale, i));
            mvInvScale.emplace_back(1.0/mvScale[i]);
        }
        img_pyramid = vector<Mat>(nlevels);
        unitNumber = N * (scale - 1) / (pow(scale, nlevels) - 1);
    }

    void computePyramid(Mat img){
        int height = img.size().height;
        int width = img.size().width;
        img_pyramid[0] = img;
        for(int i = 1; i < nlevels; i++){
            double scale = mvInvScale[i];
            Size size(width*scale, height*scale);
            resize(img_pyramid[i-1], img_pyramid[i], size, 0, 0, INTER_LINEAR);
        }
    }

    int visitNodeList(list<TreeNode> & nodes_lst, listIter iniIter, listIter finalIter, int nkp_level, std::function<listIter(listIter)> updateIter = [](listIter iter){return ++iter;}){
        vector<TreeNode> nodes_divided;
        int n = nodes_lst.size();
//        FILE *file = fopen("/home/taokun/test.dot", "a");
        for(auto it = iniIter; it != finalIter; it = updateIter(it)){
            if(it->keypoints.size() >= 2){
                nodes_divided = it->divide_nodes();
                for(auto & node:nodes_divided){
                    if(node.keypoints.size() >= 1){
                        nodes_lst.insert(it, node);
                        n++;
//                        fprintf(file, "\"(%d,%d,%d,%d)\" -> \"(%d,%d,%d,%d)\"\n", it->iniX, it->finalX, it->iniY, it->finalY, node.iniX, node.finalX, node.iniY, node.finalY);
                    }
                }
                it = nodes_lst.erase(it);
                n--;
                it--;
            }
            if(n > nkp_level){
                return n;
            }
        }
//        fclose(file);
        return n;
    }

    vector<KeyPoint> quadTreeDistributation(vector<KeyPoint> & keypoints, int nkp_level, int iniX, int finalX, int iniY, int finalY){
       TreeNode node(keypoints, iniX, finalX, iniY, finalY);
       vector<KeyPoint> result;
       list<TreeNode> nodes_lst{node};
       int n = 1;
       int last_n;
       while(n < N){
           last_n = n;
           n = visitNodeList(nodes_lst, nodes_lst.begin(), nodes_lst.end(), nkp_level);
           if(last_n == n || n >= nkp_level){
               break;
           }
           if(n + 3 * (n - last_n) >= nkp_level){
               vector<ptr2nkp> map_ptr2nkp;
               map_ptr2nkp.reserve(n + 1);
               map_ptr2nkp.emplace_back(listIter(nullptr), -1);
               for(auto it = nodes_lst.begin(); it != nodes_lst.end(); it++){
                   map_ptr2nkp.emplace_back(it, it->keypoints.size());
               }
               sort(map_ptr2nkp.begin(), map_ptr2nkp.end(), [](const ptr2nkp & A, const ptr2nkp & B){return A.nkp > B.nkp;});
               vector<ptr2nkp>::iterator vec_it = map_ptr2nkp.begin();

               auto updateIterStragety = [&vec_it](listIter it){vec_it++; listIter result = vec_it->it; return result;};

               visitNodeList(nodes_lst, map_ptr2nkp.front().it, map_ptr2nkp.back().it, nkp_level, updateIterStragety);

               break;
           }
       }

       for(auto it = nodes_lst.begin(); it != nodes_lst.end(); it++){
           auto max_it = max_element(it->keypoints.begin(), it->keypoints.end(), [](const KeyPoint &A, const KeyPoint & B){return A.response < B.response;});
           result.emplace_back(*max_it);
       }
       return move(result);
    }

    vector<KeyPoint> extractFeature(Mat img){
        computePyramid(img);
        vector<KeyPoint> keypoints;
        int width0 = img.size().width, height0 = img.size().height;
        
        for(int level = 0; level < nlevels; level++){
            Mat img_level = img_pyramid[level];
            int height = img_level.size().height, width = img_level.size().width;
            int nGridX = (width + W - 1)/W, nGridY = (height + W - 1)/W;
            int expected_nkp = N * mvInvScale[level] / (nGridX * nGridY);
            vector<KeyPoint> keypoints_level;

            for(int i = 0; i < nGridX; i++){
                for(int j = 0; j < nGridY; j++){
                    vector<KeyPoint> keypoints_subimg;
                    int iniGridX = i*W, iniGridY = j * W;
                    int finalGridX = std::min(width, iniGridX + W);
                    int finalGridY = std::min(height, iniGridY + W);
                    Mat img_sub = img_level.rowRange(iniGridY, finalGridY).colRange(iniGridX, finalGridX);
                    FAST(img_sub, keypoints_subimg, iniThred, true);
                    if(keypoints_subimg.empty()){
                        FAST(img_sub, keypoints_subimg, nextThred, true);
                    }

                    //recover the global coordinate from region coordinate.
                    for(auto & keypoint:keypoints_subimg){
                        keypoint.pt.x = (keypoint.pt.x + iniGridX) * mvScale[level];
                        keypoint.pt.y = (keypoint.pt.y + iniGridY) * mvScale[level];
                    }
                    keypoints_level.insert(keypoints_level.end(), keypoints_subimg.begin(), keypoints_subimg.end());

                }
            }
            Mat img_level_out = img.clone();

            for(auto&& keypoint:keypoints_level){
                circle(img_level_out, keypoint.pt, 2, keypoint.response, -1, 8);
            }

//            drawKeypoints(img, keypoints_level, img_level_out);
            imwrite(format("/home/taokun/undistorded-level-%d.png", level), img_level_out);

            keypoints_level = quadTreeDistributation(keypoints_level, unitNumber * mvScale[nlevels - 1 - level], 0,
                                                         width0, 0, height0);


            drawKeypoints(img, keypoints_level, img_level_out);
            imwrite(format("/home/taokun/distributed-level-%d.png", level), img_level_out);
            cout << keypoints_level.size() << endl;

            keypoints.insert(keypoints.end(), keypoints_level.begin(), keypoints_level.end());
        }
        return std::move(keypoints);
    }
};

int main(){
    cout << "hello, world" << endl;
    Mat img = imread("/home/taokun/Picture/first.jpg"), outImg;
    total_img = img.clone();
    cvtColor(img, img, COLOR_RGB2GRAY);
    ORBextractor *orb = new ORBextractor(8);
    vector<KeyPoint> kps = orb->extractFeature(img);
    drawKeypoints(img, kps, outImg);
    imwrite("/home/taokun/result.jpg", outImg);
    return 0;
}