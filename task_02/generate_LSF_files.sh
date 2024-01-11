#!/bin/bash

job_name="Batcher_MPI_job"

generate_job() {
    # $1 - proc_count
    # $2 - array_size

    LSF_filename="./lsf/"$job_name"_c"$1"_s"$2".lsf"
    touch $LSF_filename

    echo "#BSUB -n $1" >>$LSF_filename
    echo "#BSUB -W 00:15" >>$LSF_filename
    echo "#BSUB -o \"$job_name.%J.c$1.s"$2".out\"" >>$LSF_filename
    echo "#BSUB -e \"$job_name.%J.c$1.s"$2".err\"" >>$LSF_filename
    echo "#BSUB -R \"span[ptile="$((($1 + 1) / 2))"]\"" >>$LSF_filename
    echo "#BSUB -m \"polus-c3-ib polus-c4-ib\"" >>$LSF_filename
    echo "mpirun ./main "$2"" >>$LSF_filename

    echo "\tbsub < $LSF_filename"
}

rm -rf ./lsf/$job_name*

# for array_size in 100000 1000000 10000000 100000000 1000000000; do
#     for proc_count in 1 2 4 8 16 32 40; do
#         generate_job $proc_count $array_size
#     done
# done

for array_size in 100000 1000000 10000000; do
    for proc_count in 1 2 4 8 16 20; do
        generate_job $proc_count $array_size
    done
done
