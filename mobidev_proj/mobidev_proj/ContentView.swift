import SwiftUI

struct ContentView: View {
    @State private var showingHandDetection = false
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Tap Detection App")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Premi il pulsante per iniziare il riconoscimento delle mani")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
            
            Button(action: {
                showingHandDetection = true
            }) {
                HStack {
                    Image(systemName: "hand.wave.fill")
                    Text("Avvia Tap Detection")
                }
                .font(.title2)
                .foregroundColor(.white)
                .padding()
                .background(Color.blue)
                .cornerRadius(10)
            }
        }
        .padding()
        .fullScreenCover(isPresented: $showingHandDetection) {
            HandDetectionViewControllerWrapper()
        }
    }
}

// Wrapper per usare UIViewController in SwiftUI
struct HandDetectionViewControllerWrapper: UIViewControllerRepresentable {
    @Environment(\.dismiss) private var dismiss
    
    func makeUIViewController(context: Context) -> HandDetectionViewController {
        let viewController = HandDetectionViewController()
        viewController.onDismiss = {
            dismiss()
        }
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: HandDetectionViewController, context: Context) {
        // Non serve aggiornare nulla
    }
}

#Preview {
    ContentView()
}
