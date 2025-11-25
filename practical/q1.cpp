#include<bits/stdc++.h>
using namespace std;
vector<string> tokenize(const string &text) {
    vector<string> tokens;
    for (char c : text) {
        if (c != ' '&&c!=','&&c!='.')tokens.push_back(string(1, c));
    }
    return tokens;
}


map<pair<string,string>, int> count_pair_freq(const vector<string> &tokens) {
    map<pair<string,string>, int> freq;
    for(int i=0;i<(int)tokens.size();i++){
        for(int j=0;j<(int)tokens.size();j++){
            freq[{tokens[i], tokens[j]}]++;
            freq[{tokens[j], tokens[i]}]++;
        }
    }
    return freq;
}
vector<string> merge_tokens(const vector<string> &tokens,
                            const string &a, const string &b) {
    vector<string> result;
    for (int i = 0; i < (int)tokens.size();) {
        if (i+1 < tokens.size() && tokens[i] == a && tokens[i+1] == b) {
            result.push_back(a + b);
            i += 2;
            for(int j=i;j<tokens.size();j++){
                result.push_back(tokens[j]);
            }
            break;
        } else {
            result.push_back(tokens[i]);
            i++;
        }
    }
    if(result.size()!=tokens.size()-1){
        result.push_back(a+b);
    }
    // set<string> s;
    // for(auto it:result)s.insert(it);
    // result.clear();
    // for(auto it:s)resul
    return result;
}

int main(){
    string corpus ="Luna the robot woke up early today. She wanted to explore the forest near her home. The forest  was  quiet,  but  Luna  heard  a  soft  humming  sound  behind  the  old  oak  tree.    She discovered a tiny drone trying to send a distress signal. Its battery was almost empty. Luna carried  the  drone  back  to  her  workshop  and  repaired  it  using  spare  parts.    The  drone introduced itself as Pico and thanked Luna for saving it. Together, Luna and Pico built a small map of the forest using their sensors. They collected location data, sensor readings, and  signal  strength  values.  These  readings  helped  them  detect  forest  paths,  forest clearings,  and  drone-safe  zones.  By  evening, they  returned  home  with  new  data  and a plan for tomorrow's adventure. Luna stored the drone's data in her robot log, while Pico processed the map for better accuracy.";

    vector<string> tokens = tokenize(corpus);
    // for(auto it:tokens)cout<<it<<endl;
    // map<pair<string,string>,int> mp=count_pair_freq(tokens);
    // for(auto it:mp){
    //     cout<<it.first.first<<","<<it.first.second<<" "<<it.second<<endl;;
    // }
    int n = 20;
    for (int i = 0; i < n; i++) {
        map<pair<string,string>, int> freq = count_pair_freq(tokens);
        if (freq.empty()) break;
        priority_queue<pair<int, pair<string,string>>> pq;
        for (auto &p : freq) {
            pq.push({p.second, p.first});
        }
        auto best = pq.top(); pq.pop();
        int best_freq = best.first;
        string a = best.second.first;
        string b = best.second.second;
        tokens = merge_tokens(tokens, a, b);
    }
    for(auto it:tokens){
        cout<<it<<endl;
    }
    return 0;
}