//
//  signupVC.swift
//  mvobj
//
//  Created by WanYun Sun on 1/5/19.
//  Copyright © 2019 WanYun Sun. All rights reserved.
//

import UIKit
import Alamofire
import SwiftyJSON

class signupVC: UIViewController {

    
    //textFields
    @IBOutlet weak var emailTxt: UITextField!
    @IBOutlet weak var passwordTxt: UITextField!
    @IBOutlet weak var rpPasswordTxt: UITextField!
    
    
    //Buttons
    @IBOutlet weak var SignupBtn: UIButton!
    @IBOutlet weak var CancelBtn: UIButton!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        
        // Do any additional setup after loading the view.
        self.hideKeyboardWhenTappedAround()
    }
    

    @IBAction func signupBtnClick(_ sender: Any) {
        print("signup button clicked")
        //dismiss keyboard
        self.view.endEditing(true)
        
        //check if all fields are filled
        if emailTxt.text!.isEmpty || passwordTxt.text!.isEmpty || rpPasswordTxt.text!.isEmpty {
            
            //alert msg
            let alert = UIAlertController(title: "Empty field(s)", message: "Please fill all fields.", preferredStyle: UIAlertController.Style.alert)
            let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
            alert.addAction(ok)
            self.present(alert, animated: true, completion: nil)
        }
        //check if repeat password correct
        if passwordTxt.text != rpPasswordTxt.text {
            
            //alert msg
            let alert = UIAlertController(title: "Passwords not match", message: "Repeat Password has to be the same as Password.", preferredStyle: UIAlertController.Style.alert)
            let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
            alert.addAction(ok)
            self.present(alert, animated: true, completion: nil)
        }
        
        //send signup info to server
        
//        Alamofire.request("http://api.androidhive.info/contacts/").responseJSON { (responseData) -> Void in
//            if((responseData.result.value) != nil) {
//                let swiftyJsonVar = JSON(responseData.result.value!)
//                print(swiftyJsonVar)
//            }
//        }
        
    }
    
    @IBAction func cancelBtnClick(_ sender: Any) {
        print("cancel button clicked")
        self.dismiss(animated: true, completion: nil)
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

