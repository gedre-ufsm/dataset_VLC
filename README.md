# dataset_VLC
Esse programa gera um dataset dos valores IQ de um canal de comunicação VLC.


Ligue o sistema VLC do GEDRE.

Execute o arquivo complex_random_led.py.

Ao longo da execução do programa irá aparecer uma janela do GNU Radio, verifique se o gráfico apresenta um pico evidente, isso representa o valor do cross-correlation entre a entrada e a saída, caso não apareça, execute novamente. Obs: Essa janela fecha automaticamente em 5 segundos, espere.

          
Agora foram geradas 2 imagens no mesmo diretório que são 'constellation_sent.png' e constellation_received.png' que são as constelações dos dados enviados e recebidos respectivamente. O formato da constelação recebida deve ser semelhante ao enviado, mesmo que tendo uma rotação. Se aparecer apenas um formato de um disco, execute novamente o programa.


O dataset é salvo no diretório 'IQ_data' onde contém os arquivos salvos em .npy dos valores IQ que estão salvos em 3 formatos:

1 - data_complex = São os valores IQ com o número complexo (I + jQ).

IQ_x_complex = I_x + 1j * Q_x

IQ_y_complex = I_y + 1j * Q_y

                
                   
2 - data_tuple = São os valores IQ em uma tupla (1 coluna com I, outra coluna com Q).  

IQ_x_tuple = (I_x, Q_x)

IQ_y_tuple = (I_y, Q_y)


3 - data_interleaved = São os valores IQ intercalados em um único array (IQIQIQIQIQ...).

IQ_x_interleaved[0::2] = I_x

IQ_x_interleaved[1::2] = Q_x

IQ_y_interleaved[0::2] = I_y

IQ_y_interleaved[1::2] = Q_y


Desligue o sistema VLC do GEDRE após terminar. 

