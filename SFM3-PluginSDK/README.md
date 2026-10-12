# SFM3 Plugin SDK

涓?SFM Reconstructor (≥ v3.0) 寮€鍙戠畻娉曟彃浠舵墍闇€鐨勫叏閮ㄥ涓讳骇鐗┿€?
閰嶅悎 doc/鎻掍欢寮€鍙戣鑼?md 浣跨敤锛涜鑼冪 9 绔犲垪鍑哄繀椤婚伒瀹堢殑 ABI 绾㈢嚎銆?

## 鐩綍缁撴瀯

    include/sfm3/plugin/   瀹夸富鎺ュ彛澶存枃浠讹紙ISfmPlugin / IDescriptorPlugin / IMatcherPlugin /
                           ParamSchema / PluginTypes / PluginInfo锛夆€斺€斾簩杩涘埗濂戠害鐨勫０鏄庨儴鍒?
    lib/sfmcore.lib        鎺ュ彛 moc meta-object 閾炬帴搴擄紙瑙勮寖绾㈢嚎 #2锛氱姝㈣嚜琛岄噸缂栬瘧鎺ュ彛澶达級
    pri/common.pri         涓庡涓讳竴鑷寸殑缂栬瘧閰嶇疆涓庣涓夋柟璺緞锛圤penCV 460 / onnxruntime 1.26.0锛?
    templates/             瀹樻柟妯℃澘鎻掍欢锛歴ample_descriptor锛堟弿杩扮锛夈€乻ample_matcher锛堝尮閰嶅櫒锛?
    doc/鎻掍欢寮€鍙戣鑼?md     鎻掍欢寮€鍙戣鑼冿紙鎺ュ彛涓変欢濂椼€佺嚎绋嬫ā鍨嬨€丄BI 绾㈢嚎銆佹祴璇曡姹傦級
    doc/鎻掍欢寮€鍙戞祦绋?md     鎻掍欢寮€鍙戞祦绋嬶紙浠庨浂鍒板彲鍔犺浇 DLL 鐨勫垎姝ユ搷浣滄墜鍐岋級

## 蹇€熷紑濮?

1. 澶嶅埗 templates/sample_descriptor 鍒颁綘鐨勫伐浣滅洰褰曪紝鎸夎鑼冪 3/4 鑺傚疄鐜颁笁浠跺
   锛堝伐鍘?/ Worker / plugin.json锛夈€?
2. .pro 涓妸瀹夸富鍐呰矾寰勬敼涓?SDK 鐩稿璺緞锛堜袱澶勫樊寮傦級锛?

        # 瀹夸富宸ョ▼鍐呭啓娉?                    鈫?  SDK 绂荤嚎寮€鍙戝啓娉?
        INCLUDEPATH += ../../core            鈫?  INCLUDEPATH += <SDK>/include/sfm3/plugin
        LIBS += -LOUT_PWD/../../core -lsfmcore 鈫? LIBS += -L<SDK>/lib -lsfmcore

   鍏朵腑 <SDK> 寤鸿鐢ㄧ浉瀵硅矾寰勫彉閲忓紩鍏ワ紝渚嬪锛?
        SDK_ROOT = PWD/../SFM3-PluginSDK
        INCLUDEPATH += SDK_ROOT/include/sfm3/plugin
        LIBS += -LSDK_ROOT/lib -lsfmcore
   common.pri 鍚屾牱浠?SDK_ROOT/pri 寮曞叆銆?
3. 閮ㄧ讲锛氭瀯寤哄嚭鐨?DLL 涓庡叾杩愯鏃朵緷璧栨斁鍒扮殑鐩爣鏈?plugins/ 鐩綍椤讳笌瀹夸富鐗堟湰鍖归厤
   锛堝悓 Qt 6.10.1 msvc2022_64銆丮SVC 2022 /MD release銆丱penCV 460 world锛岃鑼冪孩绾?#4~#7锛夈€?
4. 楠岃瘉锛氭妸 DLL 鏀惧叆瀹夸富 plugins/ 鐩綍鍚姩瀹夸富锛屾壂鎻忔棩蹇楀嚭鐜?
   "宸插姞杞芥彃浠? <name> (<category>)" 鍗冲姞杞芥垚鍔燂紱"闈?ISfmPlugin" = IID/Q_INTERFACES 闂锛?
   "鍔犺浇澶辫触" = 杩愯鏃朵緷璧栭棶棰橈紙瑙勮寖 搂9.4锛夈€?

## 鐗堟湰鍖归厤

鏈?SDK 瀵瑰簲鐨勫涓绘瀯寤轰互 lib/sfmcore.lib 涓哄噯锛涘涓诲崌绾у悗蹇呴』浣跨敤閰嶅鐗堟湰閲嶆柊鍙戝竷
鐨?SDK 閲嶆柊缂栬瘧鎻掍欢锛圛ID 涓庤法 DLL 缁撴瀯浣撳竷灞€鏄増鏈寲濂戠害锛岃鑼?搂9.2锛夈€?
