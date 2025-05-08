while read line;do
    eval "$line"
done < config.cfg

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

./tests/test_build_roargraph \
--data_type $data_type \
--dist $metric \
--base_data_path $base_file  \
--sampled_query_data_path $train_file \
--projection_index_save_path $index_path \
--learn_base_nn_path $learn_base_nn_path \
--M_sq $M_sq --M_pjbp $M_pjbp --L_pjpq $L_pjpq -T $NumThread \
> $output_build_path &&

echo "build done"