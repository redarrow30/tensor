from cont.contact import Contacts

if __name__ == '__main__':
    # static 이면 이렇게 호출
    Contacts.run()

    # static 이 아니면 이렇게 호출
    # c = Contacts()
    # c.run()