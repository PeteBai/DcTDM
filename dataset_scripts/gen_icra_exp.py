xl_edit = {
    "B0":"A vehicle driving in-road with moderate traffic in Boston, sunny late afternoon dawn city surroundings in winter.",
    "B1":"A vehicle driving in-road with moderate traffic in Boston, sunny afternoon city surroundings in winter.",
    "B2":"A vehicle driving in-road with light traffic in Boston, night city surroundings in winter with visible snow on the side of the road.",
    "B3":"A vehicle driving in-road with moderate traffic in downtown Boston, sunny noon city surroundings in summer with skyscrapers.",
    "B4":"A vehicle driving in-road with light traffic in downtown Boston, cloudy afternoon city surroundings in winter with visible snow on the side of the road.",
    "B5":"A vehicle driving in-road with moderate traffic in Boston, sunny late afternoon city surroundings in fall.",
    "B6":"A vehicle driving in-road with moderate traffic in downtown Boston, cloudy morning city surroundings in summer.",
    "B7":"A vehicle driving in-road with moderate traffic in Boston, sunny noon city surroundings in summer.",
    "B8":"A vehicle driving in-road with light traffic in Boston, cloudy afternoon urban surroundings in spring.",
    "B9":"A vehicle driving in-road with moderate traffic in Boston, sunny noon city surroundings in summer.",
    "B10":"A vehicle driving in-road with light traffic in Boston, sunny noon city surroundings in spring.",
}

table = "1448	5185	7474	13409	19115 \
        3110	9811	20992	32314	36984 \
        4966	7144	17137	20464	24402 \
        9691	16115	17364	20132	23629 \
        3376	11108	20459	37582	40449 \
        5761	7462	12303	14754	24286 \
        6721	11371	14902	16322	19613 \
        10399	14145	17561	23875	31257 \
        4113	6540	11233	13190	15906 \
        28191	47311	53512	63902	85653 \
        759	2848	7845	9774	13682"

table_arr = table.split("         ")
# print(table_arr)

f = open("./configs/makeicraexp.txt", 'w')
for key, value in xl_edit.items():
    idx = int(key.split("B")[-1])
    line = "/work/zura-storage/Data/DSDDM_XL/depth/{}/{:06d}.npy x-x {}\n"
    for item in table_arr[idx].split('\t'):
        for cond in list(xl_edit.values()):
            linex = line.format(key, int(item), cond)
            f.write(linex)
f.close()
