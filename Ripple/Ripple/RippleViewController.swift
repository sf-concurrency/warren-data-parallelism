
import Cocoa
import MetalKit

class RippleViewController: NSViewController, MTKViewDelegate {

    var mtkView: MTKView!
    var commandQueue: MTLCommandQueue!
    
    var renderer: Renderer!
    var simulation: Simulation!
    var gridSize = MTLSizeMake(1, 1, 1)
    var mesh: MTKMesh!
    
    class func buildMesh(device: MTLDevice, gridSize:MTLSize, vertexDescriptor: MTLVertexDescriptor) -> MTKMesh? {
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        let mdlMesh = MDLMesh.newPlane(withDimensions: float2(2, 2),
                                       segments: uint2(UInt32(gridSize.width), UInt32(gridSize.height)),
                                       geometryType: .triangles,
                                       allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(vertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            return nil
        }
        attributes[0].name = MDLVertexAttributePosition
        attributes[1].name = MDLVertexAttributeTextureCoordinate
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try? MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func buildVertexDescriptor() -> MTLVertexDescriptor {
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[0].format = .float3
        mtlVertexDescriptor.attributes[0].offset = 0
        mtlVertexDescriptor.attributes[0].bufferIndex = 0
        
        mtlVertexDescriptor.attributes[1].format = .float2
        mtlVertexDescriptor.attributes[1].offset = 0
        mtlVertexDescriptor.attributes[1].bufferIndex = 1
        
        mtlVertexDescriptor.layouts[0].stride = 12
        mtlVertexDescriptor.layouts[0].stepRate = 1
        mtlVertexDescriptor.layouts[0].stepFunction = .perVertex
        
        mtlVertexDescriptor.layouts[1].stride = 8
        mtlVertexDescriptor.layouts[1].stepRate = 1
        mtlVertexDescriptor.layouts[1].stepFunction = .perVertex
        
        return mtlVertexDescriptor
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View attached to RippleViewController is not an MTKView")
            return
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        commandQueue = device.makeCommandQueue()!
        
        gridSize = MTLSize(width: 128, height: 128, depth: 1)

        mtkView.device = device
        
        mtkView.depthStencilPixelFormat = .depth32Float_stencil8
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.sampleCount = 1
        mtkView.clearColor = MTLClearColor(red: 0, green: 0.2, blue: 1.0, alpha: 1.0)

        let vertexDescriptor = RippleViewController.buildVertexDescriptor()
        
        guard let newMesh = RippleViewController.buildMesh(device: device, gridSize: gridSize, vertexDescriptor: vertexDescriptor) else {
            print("Mesh cannot be initialized")
            return
        }
        
        mesh = newMesh

        guard let newRenderer = Renderer(device: device, vertexDescriptor: vertexDescriptor) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer

        guard let newSimulation = Simulation(device: device, gridSize: gridSize) else {
            print("Renderer cannot be initialized")
            return
        }
        
        simulation = newSimulation
        
        mtkView.delegate = self
        
        renderer.drawableSize = mtkView.drawableSize
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            simulation.writeVertexPositions(to: mesh.vertexBuffers.first!.buffer,
                                            gridDimensions: gridSize,
                                            time: now,
                                            commandBuffer: commandBuffer)
            renderer.draw(mesh: mesh, in: self.view as! MTKView, commandBuffer: commandBuffer)
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawableSize = view.drawableSize
    }
    
    func addRippleCenter(_ location: NSPoint) {
        let loc = NSMakePoint(
            CGFloat(location.x) / CGFloat(view.bounds.width) * 2 - 1,
            -(CGFloat(location.y) / CGFloat(view.bounds.height) * 2 - 1))
        simulation.addRippleCenter(loc)
    }
    
    override func mouseUp(with event: NSEvent) {
        let windowLocation = event.locationInWindow
        let location = view.convert(windowLocation, from: nil)
        addRippleCenter(location)
    }
}
