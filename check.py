import client as server

# weights = [0.0, -1.4581004021186265e-12, -2.2904557539542756e-13, 4.6193416494203566e-11, -1.7523797750089007e-10, -1.81786323e-15, 8.530078243322367e-16, 2.2238718949733277e-05, -2.0500168293805344e-06, -1.6021306583209448e-08, 9.972053968100473e-10]
# weights_org = [1.0001897430147639e-18, -1.458639772701779e-12, -2.289044183029967e-13, 4.613576589138588e-11, -1.7507942070454577e-10, -1.8181053618718185e-15, 8.525220556197659e-16, 2.2267212935785478e-05, -2.0473045194467373e-06, -1.6012304250754945e-08, 9.9354894558616e-10]
# weights = [1.001e-18, -1.458e-12, -2.289e-13, 4.613576589138588e-11, -1.75e-10, -1.88e-15, 8.525220556197659e-16, 2.24e-05, -2.04842e-06, -1.6013e-08, 9.9351e-10]


# 3 2 4 6 
TEAM_ID = 'F7U8r4Y2YN0EljgonlgClHUjOIQkHpbnwCcCRi2pTo2GK1m4EZ'

# wt2 = [9.944147183366461e-19, -1.3660355749232693e-12, -2.305635082141979e-13, 4.2893991558347104e-11, -1.7567695059051306e-10, -5.724660329787282e-16, 8.310364039783746e-16, 2.9609957081268546e-05, -2.1853876228228614e-06, -1.417732345601447e-08, 9.155779917263105e-10]

wt2 = [9.94844008537157e-19, -1.3673449094175483e-12, -2.287774976787406e-13, 5.0054401512830564e-11, -1.9184139294715718e-10, -5.721186451922296e-16, 9.204875840120217e-16, 2.959497956287452e-05, -2.1821061915897267e-06, -1.3870069270276784e-08, 9.176646493561812e-10]

# wt2 = [9.943262895690264e-19, -1.3667756404995403e-12, -2.3069064189868595e-13, 4.287244934527724e-11, -1.756047668068331e-10, -5.725210559364107e-16, 8.30734040573512e-16, 2.9603057657395278e-05, -2.180539598989525e-06, -1.4172006749873707e-08, 9.175510139615484e-10]


ayush = [ 0.00000000e+00 -5.66857169e-12 ,-2.28239280e-13 , 5.04657644e-11 ,-1.91973207e-10, -1.64428324e-15 , 9.20746028e-16  ,2.83477303e-05, -2.01330090e-06, -1.60849486e-08  ,6.24279090e-10 , 1.26555592e+11 ,1.19400576e+11  ,2.60266201e+11]

# wt2 = [7.460959068240174e-19, -2.44108980394457e-12, -2.2987286683071384e-13, 4.481600647439758e-11, -1.7983257651219626e-10, -8.406302951239619e-16, 8.5417126430523275e-16, 2.9317570621508002e-05, -2.137646296914058e-06, -1.4648160675433187e-08, 8.44192024219114e-10]

# wt2 =  [9.931898113788154e-19, -1.366139985162932e-12, -2.2897779410193707e-13, 5.002609784560562e-11, -1.9194989435200715e-10, -5.717247062607806e-16, 9.197341703619821e-16, 2.9580135864444984e-05, -2.1780546213615962e-06, -1.3882479125311719e-08, 9.166141099655186e-10]
# wt2 = [9.932234867467163e-19, -1.366613938585867e-12, -2.290191426812718e-13, 4.9958388799835205e-11, -1.9190830763266981e-10, -5.718873230545639e-16, 9.2016197183686e-16, 2.9610270201431277e-05, -2.175129332208835e-06, -1.3897375856726256e-08, 8.503093033270668e-10]
# # wt2 = [9.930099439508632e-19, -1.3644458493361812e-12, -2.306762169030592e-13, 4.28913364914145e-11, -1.7559837859491334e-10, -5.722418890688463e-16, 8.310865018495532e-16, 2.959494240319699e-05, -2.177820440503937e-06, -1.417079549269045e-08, 9.178384407077303e-10]
# wt2 =  [9.93030305103278e-19, -1.3657040932080366e-12, -2.2915966376292836e-13, 4.9972141490467016e-11, -1.916963775125933e-10, -5.714352593653106e-16, 9.195233093561236e-16, 2.964531997248685e-05, -2.169137699715464e-06, -1.3869591292314848e-08, 8.514446966648333e-10]
# wt2 = [9.933469403014073e-19, -1.3647688241480805e-12, -2.29449107990652e-13, 4.997762536774338e-11, -1.9155289492118822e-10, -5.717403591794182e-16, 9.206263829793383e-16, 2.965541337875964e-05, -2.1677355509009843e-06, -1.386671526216622e-08, 8.520881789097074e-10]
# wt2 = [9.945140701021333e-19, -1.3654945182202266e-12, -2.29650238741148e-13, 4.999203151021717e-11, -1.9176316444362343e-10, -5.717124179661593e-16, 9.214487722388035e-16, 3.2174095348309117e-05, -2.127978987147717e-06, -1.3854418947295432e-08, 8.531224268409073e-10]
# wt2 = [9.94317082103698e-19, -1.3674731037605335e-12, -2.297916723070757e-13, 5.0010108794818316e-11, -1.9151405506184694e-10, -5.715122221434423e-16, 9.212991178737552e-16, 3.2206766309908164e-05, -2.1228098202600754e-06, -1.3844673865012586e-08, 8.542962399048926e-10]
# wt2 = [9.92830481746423e-19, -1.368075716414449e-12, -2.2978597724774565e-13, 5.004132395546271e-11, -1.915926012884369e-10, -5.71608307806363e-16, 9.209802623936444e-16, 3.225887429700905e-05, -2.1200192764299413e-06, -1.3829098851700823e-08, 8.56229803026645e-10]
wt2 = [9.932606888425434e-19, -1.3689208727598466e-12, -2.2953934497817487e-13, 5.004202440644622e-11, -1.914357933584885e-10, -5.714432242231101e-16, 9.20808613295086e-16, 3.228740217770987e-05, -2.1156176408472216e-06, -1.3846616239257295e-08, 8.573917714695753e-10]
# wt2 = [9.929240080359517e-19, -1.3693100885256054e-12, -2.2970410340139663e-13, 5.0035028106238e-11, -1.912605951782765e-10, -5.715292458417633e-16, 9.21095332716388e-16, 3.224037989177133e-05, -2.1141212311117358e-06, -1.3835097425363832e-08, 8.588974921711022e-10]
wt2 = [9.929240080359517e-19, -1.3693100885256054e-12, -2.2970410340139663e-13, 5.0035028106238e-11, -1.912605951782765e-10, -5.715292458417633e-16, 9.21095332716388e-16, 3.224037989177133e-05, -2.1141212311117358e-06, -1.3835097425363832e-08, 8.588974921711022e-10]

wt2 = [9.929240080359517e-19, -1.3693100885256054e-12, -2.2970410340139663e-13, 5.0035028106238e-11, -1.912605951782765e-10, -5.715292458417633e-16, 9.21095332716388e-16, 3.22037989177133e-05, -2.1141212311117358e-06, -1.3835097425363832e-08, 8.588974921711022e-10]

wt2 =  [9.917369362870062e-19, -1.3663345774890496e-12, -2.2987760470899746e-13, 5.0166628893336836e-11, -1.9137079609875392e-10, -5.707209963841367e-16, 9.219501151538645e-16, 3.2090844450778915e-05, -2.103950873147733e-06, -1.3825652924351045e-08, 8.643552922860863e-10]

status = server.submit(TEAM_ID, list(wt2))
print(status)

# for i in range(10):
# weights[7] -= 0.02e-05
train_err, valid_err = server.get_errors(TEAM_ID, list(wt2))
tot_err = train_err + valid_err
# print(wt2)
print('fitness:')
print("{:e}".format(-tot_err), "{:e}".format(train_err), "{:e}".format(valid_err))


# INITIAL_WEIGHTS = [1.0016758736507022e-18, -1.3696155264603411e-12, -2.300768584393704e-13, 4.617028826499339e-11, -1.7627848513209744e-10, -1.7730847899381538e-15, 8.38639892842589e-16, 2.2778568625222342e-05, -1.9784050209132108e-06, -1.5846641527483793e-08, 9.522475355911996e-10]


