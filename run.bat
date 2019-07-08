@ echo on
cd /d D:\PycharmProjects\1_HAN_PMF_Odin_20190517_Compare_DRMF
call activate keras35

::python han_test_0_dual_multiple.py --momentum 0.5 --pmf_lambda_u 0.001 --pmf_lambda_v 0.001 --pmf_learning_rate 0.0001 --pmf_iteration 200
::python han_test_1_user_multiple.py --momentum 0.5 --pmf_lambda_u 0.001 --pmf_lambda_v 0.001 --pmf_learning_rate 0.0001 --pmf_iteration 200
::python han_test_2_item_multiple.py --momentum 0.5 --pmf_lambda_u 0.001 --pmf_lambda_v 0.001 --pmf_learning_rate 0.0001 --pmf_iteration 200
::python cnn_gru_test_0_dual_multiple.py --momentum 0.5 --pmf_lambda_u 0.001 --pmf_lambda_v 0.001 --pmf_learning_rate 0.0001 --pmf_iteration 200
::python cnn_gru_test_1_user_multiple.py --momentum 0.5 --pmf_lambda_u 0.001 --pmf_lambda_v 0.001 --pmf_learning_rate 0.0001 --pmf_iteration 200
::python cnn_gru_test_2_item_multiple.py --momentum 0.5 --pmf_lambda_u 0.001 --pmf_lambda_v 0.001 --pmf_learning_rate 0.0001 --pmf_iteration 200
::python pmf_test_multiple.py --momentum 0.5


python han_test_0_dual_multiple.py --latent_size 5  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 25  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 50  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 75  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 100  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 125  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 150  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 175  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 200  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --latent_size 225  --datasets Musical_Instruments_5

python han_test_0_dual_multiple.py --embedding_size 30  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 60  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 90  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 120  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 150  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 180  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 210  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 240  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 270  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --embedding_size 300  --datasets Musical_Instruments_5

python han_test_0_dual_multiple.py --hidden_size 35  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 65  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 95  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 125  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 155  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 185  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 215  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 245  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 275  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --hidden_size 305  --datasets Musical_Instruments_5

python han_test_0_dual_multiple.py --han_learning_rate 0.001  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.0025  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.005  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.01  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.025  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.05  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.1  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.25  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.5  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --han_learning_rate 0.75  --datasets Musical_Instruments_5


python han_test_0_dual_multiple.py --batch_size 8  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 16  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 32  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 64  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 128  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 256  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 512  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 1024  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 2048  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --batch_size 4096  --datasets Musical_Instruments_5


python han_test_0_dual_multiple.py --pmf_learning_rate 0.001  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.0025  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.005  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.01  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.025  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.05  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.1  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.25  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.5  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --pmf_learning_rate 0.75  --datasets Musical_Instruments_5

python han_test_0_dual_multiple.py --momentum 0.1  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --momentum 0.2  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --momentum 0.3  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --momentum 0.4  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --momentum 0.5  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --momentum 0.6  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --momentum 0.7  --datasets Musical_Instruments_5
python han_test_0_dual_multiple.py --momentum 0.8  --datasets Musical_Instruments_5


pause







