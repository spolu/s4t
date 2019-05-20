(Project Goals)[https://www.notion.so/spolu/1-Pager-4a01cafdf1d842849ea7817b944c19b1]

## PROOFTRACE PPO/IOTA Distributed Training

```
mkdir -p ~/tmp/prooftrace/`git rev-parse HEAD` && prooftrace_ppo_syn_run configs/prooftrace_ppo.json --tensorboard_log_dir=~/tmp/tensorboard/`git rev-parse HEAD`_`now` --save_dir=~/tmp/prooftrace/`git rev-parse HEAD` --device=cuda:0 --sync_dir=~/tmp/iota/ppo
```

```
prooftrace_ppo_ack_run configs/prooftrace_ppo.json --sync_dir=~/tmp/iota/ppo --device=cuda:0
```

```
[20190505_0326_56.550245] ================================================
[20190505_0326_56.550245]  ProofTraces Length
[20190505_0326_56.550245] ------------------------------------------------
[20190505_0326_56.550245]      <0064 ***                              7606
[20190505_0326_56.550245]  0064-0128 *****                            13243
[20190505_0326_56.550245]  0128-0256 ********                         20294
[20190505_0326_56.550245]  0256-0512 **********                       24583
[20190505_0326_56.550245]  0512-1024 ****                             11477
[20190505_0326_56.550245]  1024-2048                                  261
[20190505_0326_56.550245]  2048-4096                                  0
[20190505_0326_56.550245]      >4096                                  0
[20190505_0326_56.550245] ------------------------------------------------
```

# note medium sqrt2

```
(z3ta) stan@syn:~/src/z3ta/data/prooftrace/medium$ ls -l test_traces/
total 1932
-rw-r--r-- 1 stan stan  10855 May  4 21:58 15037405_CUT_15037405_99_32.actions
-rw-r--r-- 1 stan stan   6714 May  4 21:58 15038203_CUT_15038203_49_19.actions
-rw-r--r-- 1 stan stan   7961 May  4 21:58 15038262_CUT_15038262_103_38.actions
-rw-r--r-- 1 stan stan   7962 May  4 21:58 15038314_CUT_15038314_103_38.actions
-rw-r--r-- 1 stan stan  10403 May  4 21:58 15038627_CUT_15038627_69_25.actions
-rw-r--r-- 1 stan stan   4630 May  4 21:57 15038708_CUT_15038708_62_21.actions
-rw-r--r-- 1 stan stan   3028 May  4 21:57 15038731_CUT_15038731_38_15.actions
-rw-r--r-- 1 stan stan   9918 May  4 21:57 15038797_CUT_15038797_125_46.actions
-rw-r--r-- 1 stan stan  24794 May  4 21:57 15039314_CUT_15039314_221_54.actions
-rw-r--r-- 1 stan stan  76883 May  4 21:57 15039514_CUT_15039514_748_156.actions
-rw-r--r-- 1 stan stan  65875 May  4 21:57 15039655_CUT_15039655_261_69.actions
-rw-r--r-- 1 stan stan  69616 May  4 21:58 15039787_CUT_15039787_242_64.actions
-rw-r--r-- 1 stan stan 104786 May  4 21:58 15039790_CUT_15039790_531_98.actions
-rw-r--r-- 1 stan stan 103718 May  4 21:58 15040438_CUT_15040438_528_98.actions
-rw-r--r-- 1 stan stan  72063 May  4 21:58 15045477_CUT_15045477_240_63.actions
-rw-r--r-- 1 stan stan  47538 May  4 21:58 15045522_CUT_15045522_182_51.actions
-rw-r--r-- 1 stan stan   4969 May  4 21:58 15045525_CUT_15045525_35_15.actions
-rw-r--r-- 1 stan stan   4950 May  4 21:59 15045529_CUT_15045529_36_9.actions
-rw-r--r-- 1 stan stan  69576 May  4 21:59 15047640_CUT_15047640_893_184.actions
-rw-r--r-- 1 stan stan  51656 May  4 21:59 15047720_CUT_15047720_319_85.actions
-rw-r--r-- 1 stan stan  87858 May  4 21:59 15047796_CUT_15047796_228_56.actions
-rw-r--r-- 1 stan stan  10874 May  4 21:59 15047866_CUT_15047866_119_38.actions
-rw-r--r-- 1 stan stan   4178 May  4 21:59 15049698_CUT_15049698_68_15.actions
-rw-r--r-- 1 stan stan   5106 May  4 21:59 15049768_CUT_15049768_41_14.actions
-rw-r--r-- 1 stan stan   8830 May  4 21:59 15049838_CUT_15049838_115_29.actions
-rw-r--r-- 1 stan stan  22298 May  4 21:59 15049906_CUT_15049906_237_48.actions
-rw-r--r-- 1 stan stan  43881 May  4 21:59 15050909_CUT_15050909_596_137.actions
-rw-r--r-- 1 stan stan  11926 May  4 21:59 15050913_CUT_15050913_152_45.actions
-rw-r--r-- 1 stan stan  29703 May  4 21:59 15051229_CUT_15051229_340_95.actions
-rw-r--r-- 1 stan stan  34576 May  4 21:59 15051233_CUT_15051233_251_79.actions
-rw-r--r-- 1 stan stan  29930 May  4 21:59 15051307_CUT_15051307_254_69.actions
-rw-r--r-- 1 stan stan 116712 May  4 21:59 15051437_CUT_15051437_626_155.actions
-rw-r--r-- 1 stan stan 157633 May  4 21:59 15051529_CUT_15051529_404_102.actions
-rw-r--r-- 1 stan stan 310259 May  4 22:00 15051613_CUT_15051613_240_60.actions
-rw-r--r-- 1 stan stan  38191 May  4 22:00 15051681_CUT_15051681_278_81.actions
-rw-r--r-- 1 stan stan  69811 May  4 22:00 15051800_CUT_15051800_547_147.actions
-rw-r--r-- 1 stan stan  38358 May  4 21:58 15051911_IRRATIONAL_SQRT_NONSQUARE_260_73.actions
-rw-r--r-- 1 stan stan  43488 May  4 21:58 15052321_IRRATIONAL_SQRT_PRIME_476_132.actions
-rw-r--r-- 1 stan stan   8502 May  4 21:58 15052391_IRRATIONAL_SQRT_2_148_45.actions
```

