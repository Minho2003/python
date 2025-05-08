drink=['생수','레몬워터','옥수수 수염차','콘트라베이스','트레비',
       '밀키스','펩시','핫식스','사이다','코코 리치','립톤','트로피카나 사과맛','트로피카나 포도맛',
       '가나','레쓰비','칸타타','카페타임','게토레이','코코 포도','식혜']
cost=[700,1600,1400,2100,1100,
      900,900,1100,1100,1100,1100,1100,1100,
      700,700,1100,1100,900,900,900]

coin_type=[100,500,1000] # 돈 종류 리스트
coin_list=[0,0,0]
coin_sum=0
def coin():  # 돈 투입, 총액 계 함수
    global coin_sum
    for i in range(0,3):
        print(str(coin_type[i])+'원의 개수를 입력하시오 : ',end='')
        coin_list[i]=int(input())
    coin_sum=100*(coin_list[0])+500*(coin_list[1])+1000*(coin_list[2])

list_index=0
def select_drink(): # 음료수 고르고 음료수의 가격 표시
    global list_index
    while True:
        drink_out=input('원하는 음료를 입력하시오 : ')
        if drink_out in drink:
            list_index=drink.index(drink_out)
            print(str(cost[list_index])+'원 입니다.')
            break
        else :
            print("없는 음료수입니다.")


cal=0
def change_coin(): # 돈 거슬러주기
    cal=(coin_sum)-(cost[list_index])
    while True:
        if cal<0:
            print("돈이 부족합니다.")
            print(str(-cal)+"원을 더 넣으십시오.")
            coin()
            cal+=coin_sum
        elif cal>=0:
            print("잔액은",str(cal)+"원 입니다.")
            break
            
coin()
select_drink()
change_coin()
#실행 시키고 coin change_coin 손보기
'''
cal=(coin_sum)-(cost[list_index])
    while 1:
        if cal<=0:
            print("돈을 더 넣으시오.")
            print(str(cal)+'원 남았습니다.')
            coin()
            print(str(coin_sum)+'원')
            
        else:
            print('잔액은 :',cal+'원 입니다.')
            break
'''

