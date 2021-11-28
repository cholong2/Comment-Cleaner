swear_list = []

def print_help():
    print(
        '',
        '----------------------------',
        '욕설 방지 프로그램',
        '0: 종료',
        '1: 금지 단어 입력',
        '2: 금지 단어 확인',
        '3: 금지 단어 삭제',
        '4: 테스트',
        '-----------------------------',
        '',
        sep='\n'
    )

while True:
    print_help()
    choice = input('실행하고자 하는 번호를 입력해주세요 >>')
    if choice == '0':
        print('프로그램을 종료합니다')
        break
    elif choice == '1':
        swear = input('입력할 단어를 입력해주세요 >>')
        if swear not in swear_list:
            swear_list.append(swear)
    elif choice == '2':
        print('금지단어 목록입니다')
        print('\n'.join(swear_list))
    elif choice == '3':
        swear = input('삭제할 단어를 입력해주세요 >>')
        if swear not in swear_list:
            print('잘못 입력했습니다')
        else:
            swear_list.remove(swear)
    elif choice == '4':
        test_string = input('테스트할 문장을 입력해주세요 >>')
        replaced_string = test_string
        for swear in swear_list:
            replaced_string = replaced_string.replace(swear, '***')
        if replaced_string == test_string:
            print('정상 문장입니다')
        else:
            print(replaced_string)
    else:
        print('잘못 입력했습니다')