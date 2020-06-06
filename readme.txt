我們要做的事大概是 
把 AIcup_testset_ok 中測資的vocal(用traverse跑1500次)
前處理過後當成input給cnn
用feature來協助cnn train 
得到onset offset 的機率
後處理轉換成[onset offset pitch]output (groundtruth的形式)
輸出成
{

    "song_id1": [

        [onset, offset, pitch],

        ...

    ],

    "song_id2": [

        [onset, offset, pitch],

        ...

    ],

    ...

}
用.json當副檔名上傳