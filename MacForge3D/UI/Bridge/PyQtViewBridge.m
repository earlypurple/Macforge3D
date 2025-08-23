#import "PyQtViewBridge.h"
#import <QtWidgets/QWidget>
#import <QtWidgets/QMainWindow>
#import <QtMacExtras/QtMacExtras>

@implementation PyQtViewBridge

+ (NSView *)createViewWithPyQtWidget:(void *)widget {
    QWidget *qwidget = (__bridge QWidget *)widget;
    if (!qwidget) {
        return nil;
    }
    
    // Créer une fenêtre Qt native
    QMainWindow *window = new QMainWindow();
    window->setCentralWidget(qwidget);
    
    // Obtenir la vue native
    NSView *view = (__bridge NSView *)window->winId();
    if (!view) {
        delete window;
        return nil;
    }
    
    // Configurer la vue
    view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    
    return view;
}

+ (void)destroyView:(NSView *)view {
    if (!view) {
        return;
    }
    
    // Récupérer la fenêtre Qt
    QMainWindow *window = (__bridge QMainWindow *)view.window;
    if (window) {
        delete window;
    }
}

@end
