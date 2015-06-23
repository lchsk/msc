#include <omp.h>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>

int
main (int argc, char* argv[])
{
    // private
    std::map<std::string, int> words_p; 
    
    // shared
    std::map<std::string, int> words_sh;
    
    // vector with all words
    std::vector<std::string> v_words;

    if (argv[1] == NULL)
    {
        std::cout << "Argument needed: text file" << std::endl;
        exit(0);
    }
    
    std::ifstream file (argv[1]);
    std::string word;
    double t;

    t = omp_get_wtime();

    if (file.is_open())
    {
        while (file >> word)
        {
            v_words.push_back(word);
        }

        file.close();
    }

    std::cout << "Time of reading file: " << (omp_get_wtime() - t) * 1000 << " ms " << std::endl;
    t = omp_get_wtime();

    int th_id, nthreads;

    std::cout << "Words: " << v_words.size() << std::endl;

    if (argv[2] == NULL)
        nthreads = omp_get_max_threads();
    else
        nthreads = atoi(argv[2]);

    std::cout << "Using " << nthreads << " threads. " << std::endl;

    int part_size = ceil(v_words.size() / (float) nthreads);

    #pragma omp sections
    {
        #pragma omp section
        {
            omp_set_num_threads(nthreads);

            #pragma omp parallel private (th_id, words_p) shared (v_words, words_sh)
            {
                th_id = omp_get_thread_num();

                int start = th_id * part_size;
                int end = start + part_size;
                int m = std::min(end, (int) v_words.size());

                for (int i = start; i < m; i++)
                {
                    std::string s = v_words[i];
                    words_p[s]++;
                }

                #pragma omp critical
                for(auto i : words_p)
                {
                    words_sh[i.first] += i.second;
                }
            }
        }
    }

    std::cout << "Time of counting: " << (omp_get_wtime() - t) * 1000 << " ms " << std::endl;

    return 0;
}