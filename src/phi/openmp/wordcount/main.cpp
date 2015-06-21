#include <omp.h>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <typeinfo>
#include <cmath>
// #include <stdio>
// #include <stdlib.h>
// #include <unistd.h>


using namespace std;

int
main (int argc, char* argv[])
{
    std::map<string, int> words;
    std::map<string, int> shwords;
    vector<string> v_words;

    ifstream file ("great-expectations.txt");
    string word;


    double start = omp_get_wtime();
    // cout << start << endl;

    if (file.is_open())
    {
        while (file >> word)
        // while (getline (myfile, line))
        {
            // cout << word << endl;
            // if (words.find(word) == words.end())
                // words[word] = 1;
            // else
                // words[word]++;
            // cout << typeid(word).name() << endl;
            v_words.push_back(word);
        }

        file.close();

    }
    else 
        cout << "Unable to open file";

    int th_id, nthreads;

    cout << "Words: " << v_words.size() << endl;
    nthreads = 4;
    int part_size = ceil(v_words.size() / (float) nthreads);
    cout << "size: " << part_size << endl;

    #pragma omp sections
    {
        #pragma omp section
        {
            omp_set_num_threads(nthreads);

            #pragma omp parallel private (th_id, words) shared (v_words, shwords)
            {
                th_id = omp_get_thread_num();
                // printf ("Thread %d\n", th_id);
                // cout << "start: " << th_id * part_size << " end: " << th_id * part_size + part_size << endl;

                int start = th_id * part_size;
                int end = start + part_size;
                int m = min(end, (int) v_words.size());
                // cout << "Min: " << m << endl;

                // cout << "s: " << start << " e: " << m << endl;
                for (int i = start; i < m; i++)
                {
                    // cout << v_words[i] << endl;
                    // if (i < v_words.size())
                        
                        // if (words.find(s) == words.end())
                        //     words[s] = 1;
                        // else
                        //     words[s]++;
                            string s = v_words[i];
                            // #pragma omp atomic
                            words[s]++;
                            // #pragma omp flush (words[s])
                            // words["do, 22"]++;

                    // words[v_words[i]]++;
                }

                #pragma omp critical
                for(auto elem : words)
                {
                    shwords[elem.first] += elem.second;
                   // std::cout << elem.first << " " << elem.second << "\n";
                }


                // if (th_id == 0)
                // {
                //     printf ("Number of threads = %d\n", omp_get_num_threads());
                // }

            }
        }
    }

    // for(auto elem : shwords)
    // {
    //    std::cout << elem.first << " " << elem.second << "\n";
    // }

    double end = omp_get_wtime();
    // cout << end << endl;

    // cout << ((double) end - (double) start) * 1000.0 / CLOCKS_PER_SEC << endl;
    cout << "Time: " << (end - start) * 1000 << endl;

    return 0;
}