#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include "../utils/ExpRecorder.h"
#include "../entities/Mbr.h"
#include "../entities/Point.h"
#include "../entities/LeafNode.h"
#include "../entities/NonLeafNode.h"
#include "../curves/z.H"
// #include "../entities/Node.h"
#include "../utils/Constants.h"
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"
#include "../entities/NodeExtend.h"
// #include "../file_utils/SearchHelper.h"
// #include "../file_utils/CustomDataSet4ZM.h"

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/nn/module.h>
// #include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
// #include <torch/nn/modules/sequential.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

using namespace std;
using namespace at;
using namespace torch::nn;
using namespace torch::optim;

class ZM
{
private:
    string file_name;
    int page_size;
    // long long side;
    int bit_num = 0;
    long long N = 0;
    long long gap;
    long long min_curve_val;
    long long max_curve_val;
    
public:
    string model_path;
    string model_path_root;
    
    ZM();
    ZM(int);

    vector<vector<std::shared_ptr<Net>>> index;

    vector<int> stages;

    vector<float> xs;
    vector<float> ys;

    int zm_max_error = 0;
    int zm_min_error = 0;
    // vector<long long> hs;

    vector<LeafNode *> leafnodes;

    // auto trainModel(vector<Point> points);
    void build(ExpRecorder &exp_recorder, vector<Point> points);

    void point_query(ExpRecorder &exp_recorder, Point query_point);
    void point_query_after_update(ExpRecorder &exp_recorder, Point query_point);
    long long get_point_index(ExpRecorder &exp_recorder, Point query_point);
    void point_query(ExpRecorder &exp_recorder, vector<Point> query_points);
    void point_query_after_update(ExpRecorder &exp_recorder, vector<Point> query_points);

    void point_query_biased(ExpRecorder &exp_recorder, Point query_point);
    void point_query_biased(ExpRecorder &exp_recorder, vector<Point> query_points);

    void window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> window_query(ExpRecorder &exp_recorder, Mbr query_window);

    void my_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> my_window_query(ExpRecorder &exp_recorder, Mbr query_window);

    void acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> acc_window_query(ExpRecorder &exp_recorder, Mbr query_windows);

    void kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);

    void my_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> my_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);

    void acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> acc_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);
    
    void my_acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> all_points, vector<Point> query_points, int k);
    vector<Point> my_acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> all_points, Point query_point, int k);

    void insert(ExpRecorder &exp_recorder, Point);
    void insert(ExpRecorder &exp_recorder, vector<Point>);

    void remove(ExpRecorder &exp_recorder, Point);
    void remove(ExpRecorder &exp_recorder, vector<Point>);
};

ZM::ZM()
{
    this->page_size = Constants::PAGESIZE;
}

ZM::ZM(int page_size)
{
    this->page_size = page_size;
}

void ZM::build(ExpRecorder &exp_recorder, vector<Point> points)
{
    auto start = chrono::high_resolution_clock::now();
    vector<vector<vector<Point>>> tmp_records;
    sort(points.begin(), points.end(), sortX());
    this->N = points.size();
    bit_num = ceil(log(N) / log(2));
    for (long i = 0; i < N; i++)
    {
        points[i].x_i = points[i].x * N;
        // xs.push_back(points[i]->x);
    }
    sort(points.begin(), points.end(), sortY());
    for (long long i = 0; i < N; i++)
    {
        points[i].y_i = points[i].y * N;
        // ys.push_back(points[i]->y);
        long long curve_val = compute_Z_value(points[i].x_i, points[i].y_i, bit_num);
        points[i].curve_val = curve_val;
    }
    sort(points.begin(), points.end(), sort_curve_val());
    min_curve_val = points[0].curve_val;
    max_curve_val = points[points.size() - 1].curve_val;
    this->gap = max_curve_val - min_curve_val;

    for (long long i = 0; i < N; i++)
    {
        // points[i]->index = i / Constants::PAGESIZE;
        points[i].index = i * 1.0 / N;
        points[i].normalized_curve_val = (points[i].curve_val - min_curve_val) * 1.0 / gap;
        // cout<< points[i]->normalized_curve_val <<endl;
    }

    int leaf_node_num = points.size() / page_size;
    // cout << "leaf_node_num:" << leaf_node_num << endl;
    for (int i = 0; i < leaf_node_num; i++)
    {
        LeafNode *leafnode = new LeafNode();
        auto bn = points.begin() + i * page_size;
        auto en = points.begin() + i * page_size + page_size;
        vector<Point> vec(bn, en);
        // cout << vec.size() << " " << vec[0]->x_i << " " << vec[99]->x_i << endl;
        leafnode->add_points(vec);
        LeafNode *temp = leafnode;
        leafnodes.push_back(temp);
    }
    exp_recorder.leaf_node_num += leaf_node_num;
    // for the last leafnode
    if (points.size() > page_size * leaf_node_num)
    {
        // TODO if do not delete will it last to the end of lifecycle?
        LeafNode *leafnode = new LeafNode();
        auto bn = points.begin() + page_size * leaf_node_num;
        auto en = points.end();
        vector<Point> vec(bn, en);
        // cout << vec.size() << " " << vec[0].x_i << " " << vec[99].x_i << endl;
        leafnode->add_points(vec);
        leafnodes.push_back(leafnode);
        exp_recorder.leaf_node_num++;
    }

    // long long N = (long long)leafnodes.size();
    stages.push_back(1);
    if (leafnodes.size() / Constants::PAGESIZE >= 4)
    {
        stages.push_back((int)(sqrt(leafnodes.size() / Constants::PAGESIZE)));
        stages.push_back(leafnodes.size() / Constants::PAGESIZE);
    }

    vector<vector<Point>> stage1;
    stage1.push_back(points);
    tmp_records.push_back(stage1);

    stage1.clear();
    stage1.shrink_to_fit();

    for (size_t i = 0; i < stages.size(); i++)
    {
        cout << "building the " << i << "st stage" << endl;
        // initialize
        vector<std::shared_ptr<Net>> temp_index;
        vector<vector<Point>> temp_points;
        int next_stage_length = 0;
        if (i < stages.size() - 1)
        {
            next_stage_length = stages[i + 1];
            for (size_t k = 0; k < next_stage_length; k++)
            {
                vector<Point> stage_temp_points;
                temp_points.push_back(stage_temp_points);
            }
            tmp_records.push_back(temp_points);
        }
        else
        {
            next_stage_length = N;//
        }
        // build
        for (size_t j = 0; j < stages[i]; j++)
        {
            model_path = model_path_root + "_" + to_string(i) + "_" + to_string(j);
            auto net = std::make_shared<Net>(1);
            #ifdef use_gpu
                net->to(torch::kCUDA);
            #endif
            if (tmp_records[i][j].size() == 0)
            {
                temp_index.push_back(net);
                continue;
            }
            try
            {
                vector<float> locations;
                vector<float> labels;
                
                std::ifstream fin(this->model_path);
                if (!fin){
                    for (Point point : tmp_records[i][j]){
                        locations.push_back(point.normalized_curve_val);
                        labels.push_back(point.index);
                    }
                    net->train_model(locations, labels);
                    torch::save(net, this->model_path);
                }
                else{
                    torch::load(net, this->model_path);
                    net->width = net->parameters()[0].sizes()[0];
                }
                net->get_parameters_ZM();
                // net->trainModel(tmp_records[i][j]);
                exp_recorder.non_leaf_node_num++;
                int max_error = 0;
                int min_error = 0;
                temp_index.push_back(net);
                for (Point point : tmp_records[i][j])
                {
                    //torch::Tensor res = net->forward(torch::tensor({point.normalized_curve_val}));
                    //int pos = 0;
                    long long pos;
                    if (i == stages.size() - 1)
                    {
                        pos = net->predict_ZM(point.normalized_curve_val) * N;
                        //pos = (int)(res.item().toFloat() * N);
                        // cout << "point->index: " << point->index << " predicted value: " << res.item().toFloat() << " pos: " << pos << endl;
                    }
                    else
                    {
                        pos = net->predict_ZM(point.normalized_curve_val) * stages[i + 1];
                        //pos = res.item().toFloat() * stages[i + 1];
                        // cout << "i: " << i << " pos: " << pos << endl;
                    }
                    if (pos < 0)
                    {
                        pos = 0;
                    }
                    if (pos >= next_stage_length)
                    {
                        pos = next_stage_length - 1;
                    }

                    if (i < stages.size() - 1)
                    {
                        tmp_records[i + 1][pos].push_back(point);
                    }
                    else
                    {
                        int error = (int)(point.index * N) - pos;
                        if (error > 0)
                        {
                            if (error > max_error)
                            {
                                max_error = error;
                            }
                        }
                        else
                        {
                            if (error < min_error)
                            {
                                min_error = error;
                            }
                        }
                    }
                }
                net->max_error = max_error;
                net->min_error = min_error;
                if ((max_error - min_error) > (zm_max_error - zm_min_error))
                {
                    zm_max_error = max_error;
                    zm_min_error = min_error;
                }
                // cout << net->parameters() << endl;
                // TODO initialize index and tmp_record
                // cout << "stage:" << i << " size:" << tmp_records[i][j].size() << endl;
                tmp_records[i][j].clear();
                tmp_records[i][j].shrink_to_fit();
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
        index.push_back(temp_index);
        // cout << "size of stages:" << stages[i] << endl;
    }

    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.max_error = zm_max_error;
    exp_recorder.min_error = zm_min_error;
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    exp_recorder.size = (1 * Constants::HIDDEN_LAYER_WIDTH + 1 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE * Constants::EACH_DIM_LENGTH + Constants::PAGESIZE * Constants::INFO_LENGTH + Constants::DIM * Constants::DIM * Constants::EACH_DIM_LENGTH) * exp_recorder.leaf_node_num;
}

void ZM::point_query(ExpRecorder &exp_recorder, Point query_point)
{
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length;
        //predicted_index = index[i][predicted_index]->predictZM(key) * next_stage_length;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    int front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    int back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    // cout << "pos: " << pos << " front: " << front << " back: " << back << " width: " << width << endl;
    while (front <= back)
    {
        int mid = (front + back) / 2;
        int node_index = mid / Constants::PAGESIZE;

        LeafNode *leafnode = leafnodes[node_index];

        if(leafnode->mbr.contains(query_point))
        {
            vector<Point>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), query_point);
            exp_recorder.page_access += 1;
            if (iter != leafnode->children->end())
            {
                //cout << "find it" << endl;
                break;
            }
        }
        if ((*leafnode->children)[0].curve_val < curve_val)
        {
            front = mid + 1;
        }
        else
        {
            back = mid - 1;
        }
        if (front > back)
        {
            //cout << "not found!" << endl;
        }
    }
}

void ZM::point_query_after_update(ExpRecorder &exp_recorder, Point query_point)
{
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length;
        //predicted_index = index[i][predicted_index]->predictZM(key) * next_stage_length;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    int front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    int back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    // cout << "predicted_index: " << predicted_index << " front: " << front << " back: " << back << endl;
    front = front / Constants::PAGESIZE;
    back = back / Constants::PAGESIZE;
    for (size_t i = front; i <= back; i++)
    {
        vector<Point>::iterator iter = find(leafnodes[i]->children->begin(), leafnodes[i]->children->end(), query_point);
        exp_recorder.page_access += 1;
        if (iter != leafnodes[i]->children->end())
        {
            // cout<< "find it!" << endl;
            break;
        }
    }
}

void ZM::point_query_biased(ExpRecorder &exp_recorder, Point query_point)
{
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length;
        //predicted_index = index[i][predicted_index]->predictZM(key) * next_stage_length;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    int front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    int back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    int mid = predicted_index;
    while (front <= back)
    {
        int node_index = mid / Constants::PAGESIZE;
        LeafNode *leafnode = leafnodes[node_index];

        if(leafnode->mbr.contains(query_point))
        {
            vector<Point>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), query_point);
            exp_recorder.page_access += 1;
            if (iter != leafnode->children->end())
            {
                //cout << "find it" << endl;
                break;
            }
        }
        if ((*leafnode->children)[0].curve_val < curve_val)
        {
            front = mid + 1;
        }
        else
        {
            back = mid - 1;
        }
        mid = (front + back) / 2;
        if (front > back)
        {
            //cout << "not found!" << endl;
        }
    }
}

void ZM::point_query(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    cout << "point_query:" << query_points.size() << endl;
    auto start = chrono::high_resolution_clock::now();
    for (long i = 0; i < query_points.size(); i++)
    {
        point_query(exp_recorder, query_points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_points.size();
    //cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    exp_recorder.page_access = exp_recorder.page_access / query_points.size();
    //cout << "finish point_query time: " << exp_recorder.time << endl;
}

void ZM::point_query_biased(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    cout << "point_query_biased:" << query_points.size() << endl;
    auto start = chrono::high_resolution_clock::now();
    for (long i = 0; i < query_points.size(); i++)
    {
        point_query_biased(exp_recorder, query_points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_points.size();
    exp_recorder.page_access = exp_recorder.page_access / query_points.size();
}

void ZM::point_query_after_update(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    auto start = chrono::high_resolution_clock::now();
    for (long i = 0; i < query_points.size(); i++)
    {
        point_query_after_update(exp_recorder, query_points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
    //cout<< "point_query_after_update time: " << exp_recorder.time << endl;
}

long long ZM::get_point_index(ExpRecorder &exp_recorder, Point query_point)
{
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    query_point.curve_val = curve_val;
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    query_point.normalized_curve_val = key;
    long long predicted_index = 0;
    long long next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }

        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length; // <====== origin
        //predicted_index = index[i][predicted_index]->predictZM(key) * next_stage_length; // <=== predict sinplely

        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    exp_recorder.index_high = predicted_index + max_error;
    exp_recorder.index_low = predicted_index + min_error;
    return predicted_index;
}

vector<Point> ZM::window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    vector<Point> window_query_results;
    vector<Point> vertexes = query_window.get_corner_points();
    vector<long long> indices;
    //cout << "-----mbr-----" << endl;
    for (Point point : vertexes)
    {
        long long predicted_index = 0;
        //cout << "x: " << point.x << " y: " << point.y << endl;
        predicted_index = get_point_index(exp_recorder, point);
        //cout << "predicted_index: " << predicted_index << endl;
        indices.push_back(exp_recorder.index_low);
        indices.push_back(exp_recorder.index_high);
        //cout << "low: " << exp_recorder.index_low << " high: " << exp_recorder.index_high << endl;
    }
    sort(indices.begin(), indices.end());
    long front = indices.front() / page_size;
    long back = indices.back() / page_size;

    front = front < 0 ? 0 : front;
    back = back >= leafnodes.size() ? leafnodes.size() - 1 : back;
    
    //cout << "front: " << front << " back: " << back << endl;
    for (size_t i = front; i <= back; i++)
    {
        LeafNode *leafnode = leafnodes[i];
        if (leafnode->mbr.interact(query_window))
        {
            exp_recorder.page_access += 1;
            for (Point point : *(leafnode->children))
            {
                if (query_window.contains(point))
                {
                    window_query_results.push_back(point);
                }
            }
        }
    }
    //cout<< window_query_results.size() <<endl;
    return window_query_results;
}

void ZM::window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    cout << "ZM::window_query" << endl;
    exp_recorder.window_query_result_size.clear();
    exp_recorder.window_query_result_size.shrink_to_fit();
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < query_windows.size(); i++)
    {
        vector<Point> window_query_result = window_query(exp_recorder, query_windows[i]);
        exp_recorder.window_query_result_size.push_back(window_query_result.size());
        //exp_recorder.window_query_results.push_back(window_query_result);  
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_windows.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_windows.size();

}

vector<Point> ZM::my_window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    vector<Point> window_query_results;
    vector<Point> vertexes = query_window.get_corner_points();
    long long predicted_index = 0;
    predicted_index = get_point_index(exp_recorder,vertexes[0]);
    long front = exp_recorder.index_low / page_size;
    front = front < 0 ? 0 : front;
    predicted_index = get_point_index(exp_recorder,vertexes[3]);
    long back = exp_recorder.index_high / page_size;
    back = back >= leafnodes.size() ? leafnodes.size() - 1 : back;
    
    //cout << "front: " << front << " back: " << back << endl;
    for (size_t i = front; i <= back; i++)
    {
        LeafNode *leafnode = leafnodes[i];
        if (leafnode->mbr.interact(query_window))
        {
            exp_recorder.page_access += 1;
            for (Point point : *(leafnode->children))
            {
                if (query_window.contains(point))
                {
                    window_query_results.push_back(point);
                }
            }
        }
    }
    return window_query_results;
}

void ZM::my_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    cout << "ZM::my_window_query" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < query_windows.size(); i++)
    {
        vector<Point> window_query_result = my_window_query(exp_recorder, query_windows[i]);
        exp_recorder.window_query_result_size.push_back(window_query_result.size());
        //exp_recorder.window_query_results.push_back(window_query_result);  
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_windows.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_windows.size();

}


vector<Point> ZM::acc_window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    vector<Point> window_query_results;
    for (LeafNode *leafnode : leafnodes)
    {
        if (leafnode->mbr.interact(query_window))
        {
            exp_recorder.page_access += 1;
            for (Point point : *(leafnode->children))
            {
                if (query_window.contains(point))
                {
                    window_query_results.push_back(point);
                }
            }
        }
    }
    // cout<< window_query_results.size() <<endl;
    return window_query_results;
}

void ZM::acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    cout << "ZM::acc_window_query" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < query_windows.size(); i++)
    {
        vector<Point> acc_window_query_result = acc_window_query(exp_recorder, query_windows[i]);
        exp_recorder.acc_window_query_result_size.push_back(acc_window_query_result.size());
        //exp_recorder.acc_window_query_results.push_back(acc_window_query_result);
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_windows.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_windows.size();
}

void ZM::kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    cout << "ZM::kNN_query" << endl;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        //exp_recorder.knn_query_results.insert(exp_recorder.knn_query_results.end(), knn_result.begin(), knn_result.end());
        exp_recorder.knn_query_results.push_back(knn_result);
        
        // cout << "knn_diff: " << knn_diff(acc_kNN_query(exp_recorder, query_points[i], k), kNN_query(exp_recorder, query_points[i], k)) << endl;
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> ZM::kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    vector<Point> result;
    float knn_query_side = sqrt((float)k / N);
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
        vector<Point> temp_result = window_query(exp_recorder, mbr);
        // cout << "mbr: " << mbr->get_self() << "size: " << temp_result.size() << endl;
        if (temp_result.size() >= k)
        {
            sort(temp_result.begin(), temp_result.end(), sortForKNN(query_point));
            Point last = temp_result[k - 1];
            // cout << " last dist : " << last->cal_dist(queryPoint) << " knnquerySide: " << knnquerySide << endl;
            if (last.cal_dist(query_point) <= knn_query_side)
            {
                auto bn = temp_result.begin();
                auto en = temp_result.begin() + k;
                vector<Point> vec(bn, en);
                result = vec;
                break;
            }
        }
        knn_query_side = knn_query_side * 2;
        // cout << " knnquerySide: " << knnquerySide << endl;
    }
    return result;
}

void ZM::my_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    cout << "ZM::my_kNN_query" << endl;
    exp_recorder.knn_query_results.clear();
    exp_recorder.knn_query_results.shrink_to_fit();
    exp_recorder.time = 0;
    exp_recorder.page_access = 0;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = my_kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        //exp_recorder.knn_query_results.insert(exp_recorder.knn_query_results.end(), knn_result.begin(), knn_result.end());
        exp_recorder.knn_query_results.push_back(knn_result);
        
        // cout << "knn_diff: " << knn_diff(acc_kNN_query(exp_recorder, query_points[i], k), kNN_query(exp_recorder, query_points[i], k)) << endl;
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> ZM::my_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    /*
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    long long curve_val2 = compute_Z_value((query_point.x + 0.01) * N, (query_point.y + 0.01) * N, bit_num);
    float key2 = (curve_val2 - min_curve_val) * 1.0 / gap;
    float delta = key2 - key;
    Point query_point2 = query_point;
    query_point2.x += 0.01;
    query_point2.y += 0.01;
    long long predict1 = get_point_index(exp_recorder,query_point);
    long long predict2 = get_point_index(exp_recorder,query_point2);
    double rho = delta / ((predict2 - predict1)*1.0/N);*/
    vector<Point> result;
    float knn_query_side = sqrt((float)k / N) * 0.25;
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
        vector<Point> temp_result = my_window_query(exp_recorder, mbr);
        // cout << "mbr: " << mbr->get_self() << "size: " << temp_result.size() << endl;
        if (temp_result.size() >= k)
        {
            sort(temp_result.begin(), temp_result.end(), sortForKNN(query_point));
            Point last = temp_result[k - 1];
            // cout << " last dist : " << last->cal_dist(queryPoint) << " knnquerySide: " << knnquerySide << endl;
            if (last.cal_dist(query_point) <= knn_query_side)
            {
                auto bn = temp_result.begin();
                auto en = temp_result.begin() + k;
                vector<Point> vec(bn, en);
                result = vec;
                break;
            }
            knn_query_side = knn_query_side * pow(2,0.5);
        }else{
            knn_query_side = knn_query_side * 2;
        }
        // cout << " knnquerySide: " << knnquerySide << endl;
    }
    return result;
}

void ZM::acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    cout << "ZM::acc_kNN_query" << endl;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = acc_kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        //exp_recorder.acc_knn_query_results.insert(exp_recorder.acc_knn_query_results.end(), knn_result.begin(), knn_result.end());
        exp_recorder.acc_knn_query_results.push_back(knn_result);
    
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.k_num = k;
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> ZM::acc_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    vector<Point> result;
    float knn_query_side = sqrt((float)k / N) * 10;
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
        vector<Point> temp_result = acc_window_query(exp_recorder, mbr);
        if (temp_result.size() >= k)
        {
            sort(temp_result.begin(), temp_result.end(), sortForKNN(query_point));
            Point last = temp_result[k - 1];
            if (last.cal_dist(query_point) <= knn_query_side)
            {
                auto bn = temp_result.begin();
                auto en = temp_result.begin() + k;
                vector<Point> vec(bn, en);
                result = vec;
                break;
            }
        }
        knn_query_side = knn_query_side * 2;
    }
    return result;
}

void ZM::my_acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> all_points,vector<Point> query_points, int k)
{
    cout << "ZM::acc_kNN_query" << endl;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = my_acc_kNN_query(exp_recorder, all_points,query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        //exp_recorder.acc_knn_query_results.insert(exp_recorder.acc_knn_query_results.end(), knn_result.begin(), knn_result.end());
        exp_recorder.acc_knn_query_results.push_back(knn_result);
    
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.k_num = k;
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> ZM::my_acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> all_points, Point query_point, int k)
{
    vector<Point> result;

    sort(all_points.begin(), all_points.end(), sortForKNN(query_point));
    auto bn = all_points.begin();
    auto en = all_points.begin() + k;
    vector<Point> vec(bn, en);

    result = vec;
    return result;
}

void ZM::insert(ExpRecorder &exp_recorder, Point point)
{
    // long long curve_val = compute_Z_value(point->x * width, point->y * width, bit_num);
    // point->curve_val = curve_val;
    // point->normalized_curve_val = (curve_val - min_curve_val) * 1.0 / gap;
    long long curve_val = compute_Z_value(point.x * N, point.y * N, bit_num);
    point.curve_val = curve_val;
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    point.normalized_curve_val = key;
    long long predicted_index = 0;
    long long next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
    std::shared_ptr<Net> *net;
    int last_model_index = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            last_model_index = predicted_index;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length;
        //predicted_index = index[i][predicted_index]->predictZM(key) * next_stage_length;
        net = &index[i][predicted_index];
        // predicted_index = net->forward(torch::tensor({key})).item().toFloat() * next_stage_length;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    exp_recorder.index_high = predicted_index + max_error;
    exp_recorder.index_low = predicted_index + min_error;

    int inserted_index = predicted_index / Constants::PAGESIZE;

    LeafNode *leafnode = leafnodes[inserted_index];

    if (leafnode->is_full())
    {
        leafnode->add_point(point);
        LeafNode *right = leafnode->split();
        leafnodes.insert(leafnodes.begin() + inserted_index + 1, right);
        index[stages.size() - 1][last_model_index]->max_error += 1;
        index[stages.size() - 1][last_model_index]->min_error -= 1;
    }
    else
    {
        leafnode->add_point(point);
    }
}

void ZM::insert(ExpRecorder &exp_recorder, vector<Point> points)
{
    cout << "ZM::insert" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        insert(exp_recorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    long long old_time_cost = exp_recorder.insert_time * exp_recorder.insert_num;
    exp_recorder.insert_num += points.size();
    //exp_recorder.insert_time = (old_time_cost + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.insert_num;
    //cout<< "insert_time: " << exp_recorder.insert_time << endl;
    exp_recorder.insert_time = (chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.insert_num;
}

void ZM::remove(ExpRecorder &exp_recorder, Point point)
{
    long long curve_val = compute_Z_value(point.x * N, point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length;
        //predicted_index = index[i][predicted_index]->predictZM(key) * next_stage_length;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    int front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    int back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    // cout << "pos: " << pos << " front: " << front << " back: " << back << " width: " << width << endl;
    back = back / Constants::PAGESIZE;
    front = front / Constants::PAGESIZE;
    for (size_t i = front; i <= back; i++)
    {
        LeafNode *leafnode = leafnodes[i];
        vector<Point>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), point);
        if (leafnode->mbr.contains(point) && leafnode->delete_point(point))
        {
            // cout << "remove it" << endl;
            break;
        }
    }

}

void ZM::remove(ExpRecorder &exp_recorder, vector<Point> points)
{
    cout << "ZM::remove" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        remove(exp_recorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    long long old_time_cost = exp_recorder.delete_time * exp_recorder.delete_num;
    exp_recorder.delete_num += points.size();
    exp_recorder.delete_time = (old_time_cost + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.delete_num;
}