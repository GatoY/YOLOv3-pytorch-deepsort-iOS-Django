//
//  loginVC.swift
//  mvobj
//
//  Created by WanYun Sun on 1/5/19.
//  Copyright © 2019 WanYun Sun. All rights reserved.
//

import UIKit
import Alamofire
import SwiftyJSON

class loginVC: UIViewController {

    //label
    @IBOutlet weak var label: UILabel!
    
    //text fileds
    @IBOutlet weak var emailTxt: UITextField!
    @IBOutlet weak var passwordTxt: UITextField!
    
    //buttons
    @IBOutlet weak var forgotPasswordBtn: UIButton!
    @IBOutlet weak var loginBtn: UIButton!
    @IBOutlet weak var signupBtn: UIButton!
    
    //variable
    var token:String = ""
    var loginURL = "http://128.250.0.207:8000/api/v1/rest-auth/login/"
    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        
        //label
        label.font = UIFont(name: "Pacifico", size: 25)
        
        
        //background
        let bg = UIImageView(frame: CGRect(x: 0,y: 0,width: self.view.frame.size.width, height: self.view.frame.size.height))
        bg.image = UIImage(named: "bg2.jpg")
        bg.layer.zPosition = -1
        self.view.addSubview(bg)
        
        self.hideKeyboardWhenTappedAround()
    }
    

    
    
    //login button clicked
    @IBAction func loginBtn_click(_ sender: Any) {
        print("log in button clicked")
        //dismiss keyboard
        self.view.endEditing(true)

        //check if all fields are filled
        if emailTxt.text!.isEmpty || passwordTxt.text!.isEmpty {

            //alert msg
            let alert = UIAlertController(title: "Empty field(s)", message: "Please fill all fields.", preferredStyle: UIAlertController.Style.alert)
            let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
            alert.addAction(ok)
            self.present(alert, animated: true, completion: nil)
            return
        }

        //create json

        let email = emailTxt.text!
        let password = passwordTxt.text!
        let params: [String: Any] = [
            "username": email,
            "email": email,
            "password": password
        ]

        //send login info to server
        AF.request(loginURL,
                   method:.post,
                   parameters: params,
                   encoding: JSONEncoding.default)

            .validate()
            .responseJSON { response in
                debugPrint(response)

                switch response.result {
                case .success (let data):
                    print("result.success")
                    //store token
                    let json = JSON(data)
                    self.token = json["token"].stringValue
                    print("key: ",self.token)

                    //call login func from AppDelegate, redirect to main page
                    let appDelegate: AppDelegate =  UIApplication.shared.delegate as! AppDelegate
                    appDelegate.login()


                case .failure(let error):
                    print("Request failed with error: \(error)")

                    //alert msg
                    let alert = UIAlertController(title: "Login Fail", message: "Please check username and password", preferredStyle: UIAlertController.Style.alert)
                    let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
                    alert.addAction(ok)
                    self.present(alert, animated: true, completion: nil)
                    return
                }
        }
    }
//                switch(response.result) {
//                case .success(_):
//                    if let data = response.result.value{
//                        print(response.result.value)
//                    }
//                    break
//
//                case .failure(_):
//                    print(response.result.error)
//                    break
//
//                }
        
//
 
    
    //forgot password clicked
    @IBAction func forgotPassword_click(_ sender: Any) {
        print("forgot password clicked")
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}

extension UIViewController {
    func hideKeyboardWhenTappedAround() {
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(hideKeyboard))
        view.addGestureRecognizer(tapGesture)
    }
    
    @objc func hideKeyboard() {
        view.endEditing(true)
    }
}

