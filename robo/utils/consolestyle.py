
## reference at http://man7.org/linux/man-pages/man4/console_codes.4.html
class Style(object):
    esc = '\x1b'
    wrap = lambda s: '\x1b[' + s + 'm'
    reset = wrap('0')
    bold = wrap('1')
    halfbright = wrap('2')
    italic = wrap('3')
    underscore = wrap('4')
    blink = wrap('5')
    reverse = wrap('7')
    
    class fg(object):
        wrap = lambda s: '\x1b[' + s + 'm'
        black = wrap('30')
        red = wrap('31')
        green = wrap('32')
        brown = wrap('33')
        blue = wrap('34')
        magenta = wrap('35')
        cyan = wrap('36')
        white = wrap('37')
        default = wrap('39')
        
    class bg(object):
        wrap = lambda s: '\x1b[' + s + 'm'
        black = wrap('40')
        red = wrap('41')
        green = wrap('42')
        brown = wrap('43')
        blue = wrap('44')
        magenta = wrap('45')
        cyan = wrap('46')
        white = wrap('47')
        default = wrap('49')

__all__ = ['Style']
