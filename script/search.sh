while read line;do
    eval "$line"
done < config.cfg

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

for i in $(seq 1 $runs); do
    if [ "$i" -eq 1 ]; then
        echo "============[$i/$runs]============" > "$output_search_path"
        ./tests/test_search_roargraph \
        --data_type $data_type \
        --dist $metric \
        --base_data_path $base_file \
        --projection_index_save_path $index_path \
        --gt_path $gt_file  \
        --query_path $query_file \
        --L_pq $L \
        --k ${topk}  -T '1' \
        >> "$output_search_path" 2>&1
    else
        echo "============[$i/$runs]============" >> "$output_search_path"
        ./tests/test_search_roargraph \
        --data_type $data_type \
        --dist $metric \
        --base_data_path $base_file \
        --projection_index_save_path $index_path \
        --gt_path $gt_file  \
        --query_path $query_file \
        --L_pq $L \
        --k ${topk}  -T '1' \
        >> "$output_search_path" 2>&1
    fi
done

echo "search done"