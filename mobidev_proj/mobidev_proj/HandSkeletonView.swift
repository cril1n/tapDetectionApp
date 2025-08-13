import UIKit
import MediaPipeTasksVision

// MARK: - Hand Skeleton Drawing View
// MARK: - Hand Skeleton Drawing View
class HandSkeletonView: UIView {

    private var points: [CGPoint] = []

    // Le connessioni tra i landmark per formare lo scheletro
    private let connections: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4),       // Pollice
        (0, 5), (5, 6), (6, 7), (7, 8),       // Indice
        (5, 9), (9, 10), (10, 11), (11, 12),  // Medio
        (9, 13), (13, 14), (14, 15), (15, 16), // Anulare
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) // Mignolo e palmo
    ]

    override func draw(_ rect: CGRect) {
        super.draw(rect)
        guard let context = UIGraphicsGetCurrentContext(), !points.isEmpty else { return }

        // Disegna le connessioni (ossa)
        context.setStrokeColor(UIColor.systemBlue.cgColor)
        context.setLineWidth(2.5)
        for connection in connections where connection.0 < points.count && connection.1 < points.count {
            let startPoint = points[connection.0]
            let endPoint = points[connection.1]
            context.move(to: startPoint)
            context.addLine(to: endPoint)
            context.strokePath()
        }

        // Disegna i landmark (giunture)
        context.setFillColor(UIColor.systemGreen.cgColor)
        for point in points {
            let landmarkRect = CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)
            context.fillEllipse(in: landmarkRect)
        }
    }

    // Funzione per aggiornare i punti e richiedere un ridisegno
    func updatePoints(_ points: [CGPoint]) {
        self.points = points
        DispatchQueue.main.async {
            self.setNeedsDisplay()
        }
    }
}
