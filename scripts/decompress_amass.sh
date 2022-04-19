###
 # @Description: 
 # @Author: JinShiyin
 # @Email: shiyinjin@foxmail.com
 # @Date: 2022-04-18 22:16:07
### 
for i in $(ls /data/jsy/datasets/AMASS/downloads/*.tar.bz2)
do
    tar -jxvf $i -C /data/jsy/datasets/AMASS/decompress
    echo "decompress [$i] done"
done