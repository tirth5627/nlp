#include<bits/stdc++.h>
using namespace std;
struct TrieNode {
    map<char, TrieNode*> children;
    bool isEndOfWord;
    int frequency; 

    TrieNode() : isEndOfWord(false), frequency(0) {}

    ~TrieNode() {
        for (auto const& [key, val] : children) {
            delete val;
        }
    }
};
class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    ~Trie() {
        delete root;
    }

    void insert(const string& word) {
        TrieNode* current = root;
        root->frequency++; 
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = new TrieNode();
            }
            current = current->children[ch];
            current->frequency++; 
        }
        current->isEndOfWord = true;
    }

    pair<string, string> find_stem_suffix(string word, bool isSuffixTrie = false) {
        if (word.empty()) {
            return {"", ""};
        }

        string processed_word = word;
        if (isSuffixTrie) {
            reverse(processed_word.begin(), processed_word.end());
        }

        TrieNode* current = root;
        TrieNode* parent = root;
        
        double max_score = -1.0;
        int split_index = -1;

        for (int i = 0; i < processed_word.length(); ++i) {
            char ch = processed_word[i];
            if (current->children.find(ch) == current->children.end()) {
                return {word, ""}; 
            }
            parent = current;
            current = current->children[ch];

            if (current->frequency > 0 && parent->frequency > 0) {
                int branching_factor = parent->children.size();
                if (branching_factor > 1) {
                     double surprise = log((double)parent->frequency / current->frequency);
                     double score = branching_factor * surprise;
                    
                    if (score > max_score) {
                        max_score = score;
                        split_index = i;
                    }
                }
            }
        }
        
        if (split_index == -1) {
            return {word, ""}; 
        }
        int final_split_pos = split_index;

        if (isSuffixTrie) {
            string rev_suffix = processed_word.substr(0, final_split_pos);
            string rev_stem = processed_word.substr(final_split_pos);
            reverse(rev_suffix.begin(), rev_suffix.end());
            reverse(rev_stem.begin(), rev_stem.end());
            return {rev_stem, rev_suffix};
        } else {
            string stem = processed_word.substr(0, final_split_pos);
            string suffix = processed_word.substr(final_split_pos);
            return {stem, suffix};
        }
    }
};


vector<string> load_words(const string& filename) {
    vector<string> words;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return words;
    }
    string word;
    while (file >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        words.push_back(word);
    }
    file.close();
    return words;
}

int main() {
    const string filename = "../ass2/brown_nouns.txt";
    vector<string> words = load_words(filename);
    Trie prefixTrie;
    Trie suffixTrie;
    for (const string& word : words) {
        prefixTrie.insert(word);
        string reversed_word = word;
        reverse(reversed_word.begin(), reversed_word.end());
        suffixTrie.insert(reversed_word);
    }

    cout << "--- Prefix Trie Stemming ---" << endl;
    for (const string& word : words) {
        auto result = prefixTrie.find_stem_suffix(word, false);
        if (!result.second.empty()) {
            cout << word << "=" << result.first << "+" << result.second << endl;
        } else {
            cout << word << "=" << result.first << endl;
        }
    }

    cout << "\n--- Suffix Trie Stemming ---" << endl;
    for (const string& word : words) {
        auto result = suffixTrie.find_stem_suffix(word, true);
        if (!result.second.empty()) {
            cout << word << "=" << result.first << "+" << result.second << endl;
        } else {
            cout << word << "=" << result.first << endl;
        }
    }

    return 0;
}