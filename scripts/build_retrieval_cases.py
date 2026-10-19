"""Freeze 100 AI-authored cases against the demo KB. Human review is still required."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.agent.rag.chunker import chunk_markdown_dir

# Five different questions per section; section groups never cross dev/test.
QUESTIONS = {
'会员权益#01-00': ['会员等级多久重新结算？','升级白银会员要消费多少？','黄金会员维持等级需要多少消费？','钻石会员的升级金额与维持金额相同吗？','会员是按今年自然年还是最近十二个月的消费计算？'],
'会员权益#02-00': ['新注册用户的无门槛券多久过期？','白银会员退货上门取件是否全免？','黄金会员每月能领什么优惠券？','钻石会员每年有几次无忧退？','钻石会员优先发货是多久出库？'],
'会员权益#03-00': ['消费一元能获得几个积分？','晒图评价额外送多少积分？','签到每天获得多少积分？','积分能抵扣整个订单的钱吗？','积分多久失效？'],
'会员权益#04-00': ['生日月份购物积分有什么变化？','谁可以领取生日礼盒？','生日礼盒是在生日当天还是当周领取？','黄金会员生日当天送什么？','白银会员生日当天能拿八十八元券吗？'],
'会员权益#05-00': ['会员要交年费吗？','能让客服手动升级会员吗？','退款成功后会影响升级累计金额吗？','付费能直接购买钻石等级吗？','会员等级能自己选择吗？'],
'常见问题FAQ#01-00': ['现货下单后多久出库？','超过七十二小时没有发货可以怎么处理？','商城支持货到付款吗？','没发货的订单能改商品颜色吗？','支付半小时还没同步要提供什么凭证？'],
'常见问题FAQ#02-00': ['钻石会员签收后价保期限是多少？','秒杀商品降价可以申请价保吗？','缺货时怎样订阅到货通知？','发现假货要保留哪些材料？','确认是假货以后按什么标准赔付？'],
'常见问题FAQ#03-00': ['寄回退货包裹丢了该找谁？','银行卡退款一般几天到账？','花呗账单已出，退款如何入账？','电器保修故障要寄到商城仓库吗？','怎样用 ORD-20240115-001 跟进售后进度？'],
'常见问题FAQ#04-00': ['忘记登录密码怎么找回？','注册手机号停用了如何申诉？','账号被盗应先做什么？','注销账户前要处理什么？','注销二十天还能恢复账号吗？'],
'退换货政策#01-00': ['签收第七天还可以无理由退货吗？','衣服已经洗过但没有破损，能无理由退吗？','刻字定制商品支持七天无理由吗？','拆封的化妆品可以无理由退货吗？','普通用户七天无理由退货的运费谁承担？'],
'退换货政策#02-00': ['商品漏发算质量售后情形吗？','质量问题签收十五天还可以退吗？','确认质量问题后退货运费谁付？','质量问题商品仓库质检通过后多久退款？','电子产品故障退货建议提供哪些证据？'],
'退换货政策#03-00': ['申请退款入口在哪里？','提交退款时要选择什么？','普通商品退款申请多久审核？','敏感商品退款审核要多久？','退款审核通过以后通常还有哪些步骤？'],
'退换货政策#04-00': ['商品还没发货可以仅退款吗？','商家承认质量问题且不要求寄回，能仅退款吗？','拒收包裹后退款大约要多久？','一单买了多件能只退一件吗？','仅退款审核通过以后多久退回原账户？'],
'配送说明#01-00': ['可以直接寄到香港吗？','想寄到海外应该从哪个入口下单？','新疆喀什以西发货是否需要额外时间？','西藏阿里额外发货时间大约是多少？','海南三沙的发货时间和普通地区相同吗？'],
'配送说明#02-00': ['广州普通快递几天到？','省会城市支持次日达吗？','县城支持当日达或者次日达吗？','预计送达时间从付款还是发货开始计算？','预售家具能按普通快递时效要求送达吗？'],
'配送说明#03-00': ['普通订单满多少包邮？','订单没满包邮门槛收多少基础运费？','家具大件按什么方式收运费？','偏远地区额外运费从多少起？','钻石会员不足九十九元的订单是否包邮？'],
'配送说明#04-00': ['可以指定用顺丰配送吗？','如何知道订单实际承运的快递公司？','商城选择快递公司的规则是什么？','商城合作快递里有没有邮政 EMS？','京东物流和圆通是否属于合作承运商？'],
'配送说明#05-00': ['贵重手机签收前应注意什么？','当面验货发现包裹破损应如何处理？','菜鸟驿站的包裹需要多久内取走？','自提柜超时不取件会怎样？','签收后才发现外观破损，多久内联系并提供什么材料？'],
'配送说明#06-00': ['第一次投递无人签收会继续配送吗？','多次投递无人签收后快递会如何处理？','物流三天不更新可以申请什么服务？','包裹错送给别人要怎样找回？','申请定位寻件需要提供什么号码？'],
}
UNKNOWN=['明年春节期间商城每天几点停止发货？','企业采购能否开增值税专用发票？','宠物食品开封后的营养损失是多少？','商城明天会上线什么优惠活动？','上海线下门店的详细地址和营业时间是什么？']

def main():
    chunks=chunk_markdown_dir(Path('app/agent/rag/knowledge'))
    byid={c.chunk_id:c for c in chunks}
    cases=[]
    for n,(key,queries) in enumerate(QUESTIONS.items()):
        assert key in byid,key
        for query in queries:
            cases.append({'id':f'rag-{len(cases)+1:03}', 'query':query,
                'group':key,'split':'dev' if n%4==0 else 'test','answerable':True,
                'gold_chunk_ids':[key], 'review_status':'ai_authored_unreviewed'})
    # Equivalent policy evidence is accepted, not penalized as a retrieval error.
    for row in cases:
        if row['query']=='钻石会员不足九十九元的订单是否包邮？':
            row['gold_chunk_ids'].append('会员权益#02-00')
    for query in UNKNOWN:
        cases.append({'id':f'rag-{len(cases)+1:03}','query':query,'group':'unknown',
            'split':'test','answerable':False,'gold_chunk_ids':[],'review_status':'ai_authored_unreviewed'})
    manifest=[{**c.to_dict(),'version':hashlib.sha256(c.text.encode()).hexdigest()} for c in chunks]
    payload={'schema_version':1,'description':'100 AI-authored demo-policy cases. Not human reviewed; not production accuracy.',
        'corpus_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'corpus':manifest,'cases':cases}
    dest=Path('app/evaluation/retrieval_cases.json')
    dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(cases)} cases / {len(chunks)} chunks -> {dest}')

if __name__=='__main__':
    main()
